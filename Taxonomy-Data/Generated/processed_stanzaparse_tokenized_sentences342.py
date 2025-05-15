import logging

# Assuming ParseResult is a class that you have defined elsewhere
# from your_module import ParseResult

def retag_tags(retag_pipeline, sentences):
    # This function should retag the sentences using the retag_pipeline
    # For demonstration, let's assume it returns a list of retagged sentences
    # You need to implement this function based on your retag_pipeline logic
    return retag_pipeline.retag(sentences)

def parse_tokenized_sentences(args, model, retag_pipeline, sentences):
    # Retag the sentences
    retagged_sentences = retag_tags(retag_pipeline, sentences)
    
    # Check if the model uses xpos
    uses_xpos = getattr(model, 'uses_xpos', False)
    
    # Create a list of words with their corresponding tags
    words_with_tags = []
    for sentence in retagged_sentences:
        words_with_tags.append([(word, word.xpos if uses_xpos else word.upos) for word in sentence])
    
    # Log a message indicating that retagging is finished
    logging.info("Retagging finished.")
    
    # Assert that the length of the words list is equal to the length of the sentences
    assert len(words_with_tags) == len(sentences), "Mismatch between words and sentences length."
    
    # Parse the sentences without gradient
    evaluation_batch_size = getattr(args, 'evaluation_batch_size', 32)  # Default batch size
    prediction_method = getattr(args, 'prediction_method', 'default')  # Default prediction method
    
    parsed_treebank = model.parse_sentences_no_grad(
        words_with_tags,
        batch_size=evaluation_batch_size,
        method=prediction_method
    )
    
    # Return the parsed treebank
    return parsed_treebank

