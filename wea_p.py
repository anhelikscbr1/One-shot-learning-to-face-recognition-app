
import weaviate

def create_db(client):
    
    client.schema.get()
    client.schema.delete_class("Img")
    class_obj = {
        "class": "Img",
        "description": "Database of one-shot learning API proyect",
        "vectorizer": "none"
    }
    client.schema.create_class(class_obj)

client = weaviate.Client(
    url="https://oneshot-learning-ugto-n5yo5ft6.weaviate.network",  # Replace with your endpoint
)
#create_db(client) #Se usa una sola vez para crear el esquema
get = client.schema.get()
#print(get)
all_objects = client.data_object.get(class_name="Img", limit =10)
object_list = all_objects.get('objects')

for img in object_list:
    print (img.get('properties').get('name'))
#results = client.query.get("Img").with_near_vector({"vector": [0.1,0.2,0.3,0.4]}).with_additional(["vector", "distance"]).do()
#print(results)