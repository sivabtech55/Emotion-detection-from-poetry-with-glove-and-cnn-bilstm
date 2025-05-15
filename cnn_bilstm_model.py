from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout

def build_model(vocab_size, embedding_dim, max_len, num_classes):
    input = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(input)
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input, outputs=output)
    return model
