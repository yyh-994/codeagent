"""In-memory SDK test double; production always uses MilvusClient."""

import copy
import json
import math

from pymilvus import MilvusClient


class FakeMilvus:
    create_schema = staticmethod(MilvusClient.create_schema)
    prepare_index_params = staticmethod(MilvusClient.prepare_index_params)

    def __init__(self):
        self.collections = {}
        self.aliases = {}
        self.last_search = None
        self.fail_upsert = False

    def _resolve(self, name):
        return self.aliases.get(name, name)

    def has_collection(self, collection_name):
        return self._resolve(collection_name) in self.collections

    def describe_collection(self, collection_name):
        name = self._resolve(collection_name)
        return {**self.collections[name]["schema"], "collection_name": name}

    def create_collection(self, collection_name, schema, index_params, consistency_level):
        self.collections[collection_name] = {"schema": schema.to_dict(), "rows": {}, "index": index_params}

    def create_alias(self, collection_name, alias):
        if alias in self.aliases:
            raise ValueError("Alias already exists")
        self.aliases[alias] = collection_name

    def alter_alias(self, collection_name, alias):
        self.aliases[alias] = collection_name

    def drop_collection(self, collection_name):
        del self.collections[collection_name]

    def upsert(self, collection_name, data):
        if self.fail_upsert:
            raise RuntimeError("Milvus unavailable")
        rows = self.collections[self._resolve(collection_name)]["rows"]
        for record in data:
            rows[record["id"]] = copy.deepcopy(record)

    def get(self, collection_name, ids, output_fields, **kwargs):
        rows = self.collections[self._resolve(collection_name)]["rows"]
        return [{field: rows[pk][field] for field in output_fields} for pk in ids if pk in rows]

    def query(self, collection_name, filter, output_fields, limit, **kwargs):
        decoder = json.JSONDecoder()
        remainder = filter.removeprefix("type == ")
        kind, end = decoder.raw_decode(remainder)
        content = decoder.decode(remainder[end:].removeprefix(" and content == "))
        rows = self.collections[self._resolve(collection_name)]["rows"]
        return [{field: row[field] for field in output_fields} for row in rows.values()
                if row["type"] == kind and row["content"] == content][:limit]

    def query_iterator(self, collection_name, output_fields, **kwargs):
        rows = self.collections[self._resolve(collection_name)]["rows"]
        batches = [[{field: row[field] for field in output_fields} for row in rows.values()], []]

        class Iterator:
            def next(self):
                return batches.pop(0)

            def close(self):
                pass

        return Iterator()

    def delete(self, collection_name, ids):
        rows = self.collections[self._resolve(collection_name)]["rows"]
        for pk in ids:
            rows.pop(pk, None)

    def search(self, collection_name, data, limit, output_fields, **kwargs):
        self.last_search = {"collection_name": collection_name, "data": data,
                            "limit": limit, "output_fields": output_fields, **kwargs}
        rows = self.collections[self._resolve(collection_name)]["rows"]
        query = data[0]
        results = []
        for row in rows.values():
            vector = row["vector"]
            score = sum(a * b for a, b in zip(query, vector)) / (math.hypot(*query) * math.hypot(*vector))
            results.append({"id": row["id"], "distance": score,
                            "entity": {field: row[field] for field in output_fields}})
        return [sorted(results, key=lambda item: -item["distance"])[:limit]]

    def close(self):
        pass
