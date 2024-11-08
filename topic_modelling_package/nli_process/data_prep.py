from pyspark import DataFrame

def parse_json_list(s):
    
    try:
        return json.loads(s)
    except Exception as e:
        logger.error(f"JSON parsing error: {e} for input: {s}")
        return []

def create_text_pairs(filt, labels):
    text_pairs = []
    for t in filt:
        for l in labels:
            text_pairs.append(f"{t}</s></s>{l}.")
    return text_pairs

