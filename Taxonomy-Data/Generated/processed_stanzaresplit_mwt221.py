import stanza
from stanza.models.common.doc import Document

def resplit_mwt(tokens, pipeline, keep_tokens):
    # Check if the pipeline contains the necessary processors
    if 'tokenize' not in pipeline.processors or 'mwt' not in pipeline.processors:
        raise ValueError("The pipeline must contain both 'tokenize' and 'mwt' processors.")
    
    # Create a Document object from the tokens
    # Each sentence in the tokens list should be a list of strings (tokens)
    doc = Document([], text=None)
    for sentence in tokens:
        doc.add_sentence(sentence)
    
    # Use the tokenize processor to predict token boundaries
    tokenized_doc = pipeline.processors['tokenize'].predict(doc)
    
    # If keep_tokens is True, enforce the old token boundaries
    if keep_tokens:
        for i, sentence in enumerate(tokenized_doc.sentences):
            for j, token in enumerate(sentence.tokens):
                # Replace the predicted tokens with the original tokens
                token.words = [stanza.models.common.doc.Word(text=tokens[i][j], id=j+1)]
    
    # Process the document using the mwt processor
    mwt_doc = pipeline.processors['mwt'].process(tokenized_doc)
    
    return mwt_doc

