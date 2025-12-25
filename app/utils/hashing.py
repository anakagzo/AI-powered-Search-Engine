import hashlib

def deterministic_hash(value: str) -> str:
    """
    Generates a deterministic SHA-256 hash for a given string value. 
    

    args:
        value (str): The input string to hash.  
        returns: The SHA-256 hash of the input string.
    """
    return hashlib.sha256(value.encode()).hexdigest()
