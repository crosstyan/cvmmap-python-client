import errno
import importlib
import mmap
import os
import threading
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from typing import Any, cast


if os.name != "posix":
    raise RuntimeError("cvmmap SharedMemory helper currently supports POSIX only")


_posixshmem = cast(Any, importlib.import_module("_posixshmem"))
_USE_POSIX = True


class SharedMemory:
    _name: str | None = None
    _fd: int = -1
    _mmap: mmap.mmap | None = None
    _buf: memoryview | None = None
    _size: int = 0
    _track: bool
    _readonly: bool
    _prepend_leading_slash: bool = True if _USE_POSIX else False
    __lock = threading.Lock()

    def __init__(
        self,
        name: str | None = None,
        create: bool = False,
        size: int = 0,
        *,
        track: bool = True,
        readonly: bool = False,
    ) -> None:
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create and readonly:
            raise ValueError("readonly=True is incompatible with create=True")
        if name is None and not create:
            raise ValueError("'name' can only be None if create=True")

        self._track = track
        self._readonly = readonly
        flags = os.O_RDONLY if readonly and not create else os.O_RDWR
        if create:
            flags |= os.O_CREAT | os.O_EXCL
            if size == 0:
                raise ValueError("'size' must be a positive number different from zero")

        if name is None:
            if readonly:
                raise ValueError("readonly shared memory requires an explicit name")
            stdlib_shm = _mpshm.SharedMemory(name=None, create=True, size=size)
            self._adopt_stdlib_instance(stdlib_shm, track=track)
            return

        shm_name = "/" + name if self._prepend_leading_slash else name
        self._name = shm_name

        register_name = self._name
        register_enabled = track
        with self.__lock:
            orig_register = _mprt.register
            if not track:
                _mprt.register = self.__tmp_register
            try:
                self._fd = int(_posixshmem.shm_open(shm_name, flags, mode=0o600))
                if create and size:
                    os.ftruncate(self._fd, size)
                stats = os.fstat(self._fd)
                self._size = int(stats.st_size)
                access = (
                    mmap.ACCESS_READ if readonly and not create else mmap.ACCESS_WRITE
                )
                self._mmap = mmap.mmap(self._fd, self._size, access=access)
            except OSError:
                self.close()
                if create:
                    try:
                        _posixshmem.shm_unlink(shm_name)
                    except OSError as unlink_exc:
                        if unlink_exc.errno != errno.ENOENT:
                            raise
                raise
            finally:
                _mprt.register = orig_register

        if register_enabled and register_name is not None:
            _mprt.register(register_name, "shared_memory")

        self._buf = memoryview(self._mmap)

    def _adopt_stdlib_instance(self, shm: _mpshm.SharedMemory, *, track: bool) -> None:
        self._name = getattr(shm, "_name", None)
        self._fd = int(getattr(shm, "_fd", -1))
        self._mmap = cast(mmap.mmap | None, getattr(shm, "_mmap", None))
        self._buf = cast(memoryview | None, getattr(shm, "_buf", None))
        self._size = int(getattr(shm, "_size", 0))
        self._track = track
        self._readonly = False
        setattr(shm, "_fd", -1)
        setattr(shm, "_mmap", None)
        setattr(shm, "_buf", None)
        setattr(shm, "_name", None)

    @staticmethod
    def __tmp_register(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        return

    @property
    def buf(self) -> memoryview:
        if self._buf is None:
            raise ValueError("operation on closed shared memory")
        return self._buf

    @property
    def name(self) -> str:
        if self._name is None:
            raise ValueError("shared memory name is unavailable")
        if _USE_POSIX and self._prepend_leading_slash and self._name.startswith("/"):
            return self._name[1:]
        return self._name

    @property
    def size(self) -> int:
        return self._size

    def close(self) -> None:
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if _USE_POSIX and self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def unlink(self) -> None:
        if _USE_POSIX and self._name:
            _posixshmem.shm_unlink(self._name)
            if self._track:
                _mprt.unregister(self._name, "shared_memory")

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass
