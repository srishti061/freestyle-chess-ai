import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import zstandard as zstd
import io
from tensorflow.keras.models import load_model


def load_games(pgn_path, num_games=10000):
    games = []
    with open(pgn_path, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(compressed)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

        for _ in range(num_games):
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            games.append(game)
    return games


def create_move_lookup(games):
    move_vocab = set()
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            move_vocab.add(move.uci())
            board.push(move)
    return {move: i for i, move in enumerate(sorted(move_vocab))}


def board_to_tensor(board):
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            channel = (piece.piece_type - 1) * 2 + (0 if piece.color else 1)
            tensor[row, col, channel] = 1.0
    return tensor


def create_model(input_shape, output_size):
    model = models.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(output_size, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


class ChessDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, games, move_lookup, batch_size=32):
        self.games = games
        self.move_lookup = move_lookup
        self.batch_size = batch_size
        self.positions = []

        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                self.positions.append((board.copy(), move))
                board.push(move)

    def __len__(self):
        return int(np.ceil(len(self.positions) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.positions[idx * self.batch_size : (idx + 1) * self.batch_size]
        X = np.array([board_to_tensor(b) for b, _ in batch])
        y = np.array([self.move_lookup[m.uci()] for _, m in batch])
        return X, y


class MLAI:
    def __init__(self, model, move_lookup):
        self.model = model
        self.move_lookup = move_lookup
        self.reverse_lookup = {v: k for k, v in move_lookup.items()}

    def predict_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        input_tensor = board_to_tensor(board)
        predictions = self.model.predict(np.expand_dims(input_tensor, 0))[0]

        move_probs = []
        for move in legal_moves:
            uci = move.uci()
            if uci in self.move_lookup:
                idx = self.move_lookup[uci]
                move_probs.append((move, predictions[idx]))
            else:
                move_probs.append((move, 0.0))

        return max(move_probs, key=lambda x: x[1])[0]


if __name__ == "__main__":
    import os

    games = load_games("lichess.pgn.zst")
    move_lookup = create_move_lookup(games)
    generator = ChessDataGenerator(games, move_lookup)

    if os.path.exists("chess_model.h5"):
        print("Loading existing model...")
        model = load_model("chess_model.h5")
    else:
        print("Training new model...")
        model = create_model((8, 8, 12), len(move_lookup))
        model.fit(generator, epochs=10)
        model.save("chess_model.h5")
        print("Model saved successfully!")

    # 👉 This should ALWAYS run (not inside else)
    ai = MLAI(model, move_lookup)

    board = chess.Board()
    while not board.is_game_over():
        print(board)
        move = ai.predict_move(board)
        board.push(move)
        print(f"AI move: {move}")
        print("---")

    print("Game Over")
    print(f"Result: {board.result()}")
