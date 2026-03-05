import asyncio
import click
import zmq
import zmq.asyncio


async def test_sync_reception(instance_name: str):
    addr = f"ipc:///tmp/cvmmap_{instance_name}"
    print(f"Connecting to SUB socket at {addr}")

    ctx = zmq.asyncio.Context.instance()
    sock = ctx.socket(zmq.SUB)

    # We want everything to just see if anything is coming through
    sock.subscribe(b"")
    sock.connect(addr)

    print("Waiting for messages... (Ctrl+C to stop)")
    try:
        while True:
            # Just wait and see if we get ANY bytes
            msg = await sock.recv()

            if len(msg) > 0:
                magic = msg[0]
                magic_hex = hex(magic)
                # 0x7d is FRAME_TOPIC_MAGIC, 0x5a is MODULE_STATUS_MAGIC based on msg.py
                topic_type = (
                    "FRAME_SYNC"
                    if magic == 0x7D
                    else ("MODULE_STATUS" if magic == 0x5A else "UNKNOWN")
                )
                print(
                    f"Received message of {len(msg)} bytes. Magic: {magic_hex} ({topic_type})"
                )
            else:
                print("Received empty message")

    except asyncio.CancelledError:
        print("Stopped.")
    finally:
        sock.close()


@click.command()
@click.argument("instance_name", default="default")
def cli(instance_name: str):
    try:
        asyncio.run(test_sync_reception(instance_name))
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    runner = getattr(cli, "main", None)
    if callable(runner):
        runner(standalone_mode=True)
