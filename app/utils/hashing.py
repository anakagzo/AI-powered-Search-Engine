import hashlib

def deterministic_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
