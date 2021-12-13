import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers, ErrConnectionClosed
import json
from datetime import datetime


class NatsClient:

    def __init__(self):
        self._url = 'your url'
        self._topic = 'Coffee.core.detection'
        self._nc = NATS()

    async def send_msg(self, message, event_loop):
        """Request message to self.topic"""

        try:
            await self._nc.connect(servers=[self._url], loop=event_loop)
        except (ErrNoServers, ErrTimeout) as err:
            print(err)

        try:
            answer = await self._nc.request(self._topic, json.dumps(message).encode())
            data = answer.data.decode()
            print(data)
        except ErrTimeout as err:
            print(err)


    async def receive_msg(self, event_loop):
        """Receive message from self.topic"""

        try:
            await self._nc.connect(servers=[self._url], loop=event_loop)
        except (ErrNoServers, ErrTimeout) as err:
            print(err)

        async def _receive_callback(msg):
            data = json.loads(msg.data.decode())
            print(data)
            reply = 'YOUR REPLY HERE'
            await self._nc.publish(msg.reply, json.dumps(reply).encode())


        await self._nc.subscribe(self._topic, cb=_receive_callback)
        await self._nc.flush()


if __name__ == '__main__':

    loop = asyncio.get_event_loop()
    loop.create_task(NatsClient().send_msg('Hello, World!', loop))
    
    loop.run_forever()
    loop.close()
