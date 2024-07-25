from verbs.graphrag import *


def create_final_documents(base_documents_output):
    final_documents_output = rename(
        base_documents_output, **{"columns": {"text_units": "text_unit_ids"}}
    )
    return final_documents_output
