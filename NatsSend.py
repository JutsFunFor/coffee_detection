import asyncio

from NatsClient import NatsClient

if __name__ == '__main__':
    client = NatsClient()
    loop = asyncio.get_event_loop()
    loop.create_task(client.send_msg('Hello, World!', loop))

    try:
        loop.run_forever()
    except Exception as err:
        print(err)
    finally:
        loop.close()