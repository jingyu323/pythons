import random
import paho.mqtt.client as mqtt

# The callback function of connection
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("python/mqtt")

# The callback function for received message
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))


client_id = f'python_mqtt_{random.randint(0, 1000)}'
client = mqtt.Client(client_id)
client.on_connect = on_connect
client.on_message = on_message
client.connect("192.168.99.179", 1883, 60)
client.loop_forever()
