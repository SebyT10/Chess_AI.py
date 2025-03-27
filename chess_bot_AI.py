import chess
import chess.engine
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys

# Define the neural network for policy and value prediction
class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, 64)  # Move probabilities
        self.value_head = nn.Linear(128, 1)    # Board evaluation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        policy = self.softmax(self.policy_head(x))  # Move probabilities
        value = self.tanh(self.value_head(x))  # Board evaluation between -1 and 1
        return policy, value

# Convert a board state to a tensor
def board_to_tensor(board):
    board_str = str(board).replace(".", "0").replace(" ", "").replace("\n", "")
    mapping = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
               "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6, "0": 0}
    board_array = [mapping[char] for char in board_str]
    return torch.tensor(board_array, dtype=torch.float32)

# Monte Carlo Tree Search (MCTS)
def mcts(board, model, simulations=100):
    legal_moves = list(board.legal_moves)
    move_scores = np.zeros(len(legal_moves))
    
    for _ in range(simulations):
        for i, move in enumerate(legal_moves):
            board.push(move)
            with torch.no_grad():
                _, value = model(board_to_tensor(board).unsqueeze(0))
            move_scores[i] += value.item()
            board.pop()
    
    best_move = legal_moves[np.argmax(move_scores)]
    return best_move

# Play against the AI
model = ChessNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load model if available
if os.path.exists("chess_model.pth"):
    model.load_state_dict(torch.load("chess_model.pth"))
    print("Loaded trained model.")

def play():
    board = chess.Board()
    print("Welcome to Chess AI! Enter moves in UCI format (e.g., e2e4)")
    
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            move = input("Your move: ")
            board.push(chess.Move.from_uci(move))
        else:
            ai_move = mcts(board, model, simulations=100)
            board.push(ai_move)
            print(f"AI plays: {ai_move}")
    
    print("Game Over! Result:", board.result())

# Train from PGN dataset
def train_from_pgn(pgn_file, num_games=1000):
    with open(pgn_file) as f:
        for _ in range(num_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                reward = 1 if board.is_checkmate() else 0.1 if board.is_check() else 0
                train(board, reward)
    print("Training from PGN completed!")

# Training function
def train(board, reward):
    model.train()
    optimizer.zero_grad()
    input_tensor = board_to_tensor(board).unsqueeze(0)
    predicted_policy, predicted_value = model(input_tensor)
    target_value = torch.tensor([[reward]], dtype=torch.float32)
    loss_fn = nn.MSELoss()
    loss = loss_fn(predicted_value, target_value)
    loss.backward()
    optimizer.step()
    
    # Save model after training step
    torch.save(model.state_dict(), "chess_model.pth")

# Self-play training loop
def self_play(num_games=1000):
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            ai_move = mcts(board, model, simulations=100)
            board.push(ai_move)
            reward = 1 if board.is_checkmate() else 0.1 if board.is_check() else 0
            train(board, reward)
    print("Self-play training completed!")

# Uncomment to play against the AI
# play()

# Uncomment to train from a PGN dataset
# train_from_pgn("grandmaster_games.pgn", num_games=1000)

# Uncomment to train via self-play
self_play(num_games=5)

print("Training complete. Exiting...")
sys.exit(0)