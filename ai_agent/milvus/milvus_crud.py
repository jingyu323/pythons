from pymilvus import MilvusClient

# ============ 1. 连接 Milvus ============
client = MilvusClient(
    uri="http://192.168.182.128:19530",
  
    db_name="default",
)

# ============ 2. 准备集合 ============
collection_name = "products"

if client.has_collection(collection_name):
    client.drop_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    dimension=5,
    primary_field_name="id",
    id_type="int",
    metric_type="COSINE",
)

print("集合已创建")

# ============ 3. 插入 Entity ============
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4, 0.5], "name": "A"},
    {"id": 2, "vector": [0.2, 0.3, 0.4, 0.5, 0.6], "name": "B"},
    {"id": 3, "vector": [0.3, 0.4, 0.5, 0.6, 0.7], "name": "C"},
]

insert_res = client.insert(collection_name=collection_name, data=data)
print("插入结果:", insert_res)

# ============ 4. 查询 Entity ============
query_res = client.query(
    collection_name=collection_name,
    filter="id in [1, 2]",
    output_fields=["id", "name", "vector"],
)
print("查询结果:", query_res)

# ============ 5. 向量检索 ============
search_results = client.search(
    collection_name=collection_name,
    data=[[0.15, 0.25, 0.35, 0.45, 0.55]],
    limit=2,
    output_fields=["id", "name"],
)

print("检索结果:")
for hit in search_results[0]:
    print(hit)

# ============ 6. 更新 Entity ============
upsert_data = [
    {"id": 1, "vector": [0.11, 0.22, 0.33, 0.44, 0.55], "name": "A_updated"},
]

client.upsert(collection_name=collection_name, data=upsert_data)
print("upsert 完成")

# 确认更新结果
updated = client.query(
    collection_name=collection_name,
    filter="id == 1",
    output_fields=["id", "name"],
)
print("更新后查询:", updated)

# ============ 7. 删除 Entity ============
client.delete(collection_name=collection_name, ids=[3])
print("删除 id=3 完成")

# 确认删除结果
remaining = client.query(
    collection_name=collection_name,
    filter="id > 0",
    output_fields=["id", "name"],
)
print("剩余数据:", remaining)

# ============ 8. 清理 ============
client.drop_collection(collection_name)
print("集合已删除，脚本执行完毕")
