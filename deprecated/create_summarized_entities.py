from verbs.summarise_descriptions import summarize_descriptions


def create_summarized_entities(dataset, send_to):
    dataset = summarize_descriptions(
        dataset,
        send_to=send_to,
        column="entity_graph",
        to="entity_graph",
        strategy={
            "llm": {
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "max_summary_length": 500,
        },
    )

    return dataset
