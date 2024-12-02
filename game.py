# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:43:17 2024

@author: geo_a
"""

import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import load_model

import gym
from gym import spaces

class Player:
    def __init__(self, player_id):
        self.player_id = player_id  # Identifier for the player (1 or 2)
        self.hand = []  # Cards in the player's hand
        self.score = 0  # Player's current score
        self.round_wins = 0 # how many rounds the player has won
        self.mulligans_left = 3  # Mulligans left, adjustable per round
        self.passed = False  # Whether the player has passed in the current round
        self.cards_played_this_round = []
        self.cards_played = []
        
    def reset(self):
        #self.player_id = player_id  # Identifier for the player (1 or 2)
        self.hand = []  # Cards in the player's hand
        self.score = 0  # Player's current score
        self.round_wins = 0 # how many rounds the player has won
        self.mulligans_left = 3  # Mulligans left, adjustable per round
        self.passed = False  # Whether the player has passed in the current round
        self.cards_played_this_round = []
        self.cards_played = []
    
    def draw_card(self, deck):
        """Draw a single card from the deck and add it to the player's hand."""
        if deck:
            card = deck.pop(0)  # Assuming the deck is a list with the top card at index 0
            self.hand.append(card)

    def play_card(self, card_value):
        """Play a card by its value."""
        if card_value in self.hand:
            self.hand.remove(card_value)
            self.score += card_value  # Assuming the score is directly influenced by the card's value
            self.cards_played_this_round.append(card_value)
            self.cards_played.append(card_value)
            return True
        return False

    def reset_for_new_round(self):
        """Resets the player's state for a new round."""
        self.score = 0
        self.passed = False
        self.cards_played_this_round.clear()  # Clear the list of played cards for the new round

    def perform_mulligan(self, card_value, player_deck, buffer_deck):
        if self.mulligans_left > 0 and card_value in self.hand:
            self.hand.remove(card_value)
            buffer_deck.append(card_value)  # Add the mulliganed card to the buffer deck
            self.mulligans_left -= 1
            
            if player_deck:  # Check if the deck has cards left
                # Draw a new card from the player's own deck
                new_card = player_deck.pop(0)
                self.hand.append(new_card)
                return True  # Mulligan successful
            else:
                return False  # Deck empty, mulligan not fully successful
        return False  # Mulligan not successful due to other conditions
    
    def pass_turn(self):
        """Mark the player as having passed for the current round."""
        self.passed = True



class Game:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.decks = [list(range(1, 17)), list(range(1, 17))]  # Initialize separate decks for each player
        self.game_over = False
        self.game_phase = 0 # 0 for mulligan phase, 1 for actual round phase
        self.current_round = 1
        self.last_round_winner = None
        self.buffer_deck = []  # To hold mulliganed cards temporarily
        self.played_cards_history = []  # Initialize an empty list to store history
        self.winner = None
        
    def reset(self):
        #self.players = [player1, player2]
        self.decks = [list(range(1, 17)), list(range(1, 17))]  # Initialize separate decks for each player
        self.game_over = False
        self.game_phase = 0 # 0 for mulligan phase, 1 for actual round phase
        self.current_round = 1
        self.last_round_winner = None
        self.buffer_deck = []  # To hold mulliganed cards temporarily
        self.played_cards_history = []  # Initialize an empty list to store history
        self.winner = None
        
        for player in self.players:
            player.reset()        
        
    def shuffle_decks(self):
        """Shuffles the decks at the beginning of the game."""
        for deck in self.decks:
            random.shuffle(deck)
    
    def draw_hand(self):
        """Draws cards for each player from their respective decks, ensuring the hand size does not exceed 10 cards."""
        cards_to_draw = 10 if self.current_round == 1 else 3  # 10 cards for the first round, 3 for subsequent rounds
        
        for i, player in enumerate(self.players):
            # Determine the actual number of cards to draw, considering the maximum hand size of 10 cards
            actual_cards_to_draw = min(cards_to_draw, 10 - len(player.hand))
            
            for _ in range(actual_cards_to_draw):
                if self.decks[i]:  # Check if the player's deck has cards left
                    card = self.decks[i].pop(0)  # Draw the top card from the player's deck
                    player.hand.append(card)
            # Optional: Sort the player's hand if required by your game's rules
            player.hand.sort()

    def start_round_mulligan_phase(self):
        """Allows each player to perform mulligans at the start of a round."""
        
        self.game_phase=1
        
        # Reset mulligans based on the rules mentioned
        if self.current_round ==1:
            self.players[0].mulligans_left = 3  # Player 1 has 3 mulligans
            self.players[1].mulligans_left = 2  # Player 2 has 2 mulligans
        else: 
            self.players[0].mulligans_left = 2  # Player 1 has 2 mulligans
            self.players[1].mulligans_left = 2  # Player 2 has 2 mulligans


        for i, player in enumerate(self.players):
            print(f"Player {i + 1}'s turn to mulligan.")
            
            if not self.decks[i]:  # If the deck is empty
                print(f"Player {i + 1}'s deck is empty, skipping mulligan phase.")
                continue  # Skip to the next player
            
            if isinstance(player, AIPlayer):
                while player.mulligans_left > 0:
                    
                    vgs = self.get_visible_game_state(player.player_id)
                    
                    nn_input = self.prepare_nn_input(player.player_id)
                    nn_input_formatted = np.expand_dims(nn_input, axis=0)
                    model_output = player.model.predict(nn_input_formatted)[0]  # Assuming model is accessible here
                    
                    # Get valid actions for mulligans, similar to playing phase but focused on cards in hand
                    valid_actions_mask = self.get_valid_actions(vgs)
                    masked_output = model_output * valid_actions_mask
                    selected_action = np.argmax(masked_output)
            
                    if selected_action < 16:  # Assuming indices 0-15 correspond to mulliganing cards 1-16
                        card_to_mulligan = selected_action + 1
                        successful_mulligan = player.perform_mulligan(card_to_mulligan, self.decks[i], self.buffer_deck)
                        if successful_mulligan:
                            print(f"AI Player {player.player_id} mulliganed card {card_to_mulligan}.")
                            if not self.decks[i]:  # If the deck becomes empty
                                print(f"Player {i + 1}'s deck is now empty, ending mulligan phase.")
                                break
                        else:
                            print(f"AI Player {player.player_id} attempted an invalid mulligan.")
                    else:
                        # Pass the mulligan phase if the AI decides to pass or if mulligan is not possible
                        print(f"AI Player {player.player_id} has passed their mulligan phase.")
                        break

            else:
                while player.mulligans_left > 0:
                    print(f"Player {i + 1} hand: {player.hand}")
                    mulligan = input("Choose a card to mulligan or type 'pass' to finish: ").strip()
                    
                    if mulligan.lower() == 'pass':
                        print(f"Player {i + 1} has passed their mulligan phase.")
                        break  # Exit the mulligan loop for this player
                    
                    try:
                        mulligan_card = int(mulligan)
                        if mulligan_card in player.hand:
                            # Perform the mulligan logic
                            successful_mulligan = player.perform_mulligan(mulligan_card, self.decks[i], self.buffer_deck)
                            if successful_mulligan: 
                                print(f"Player {i + 1} mulliganed card {mulligan_card}.")
                                if not self.decks[i]:  # If the deck becomes empty
                                    print(f"Player {i + 1}'s deck is now empty, ending mulligan phase.")
                                    break
                            else:
                                print("Deck is empty. No more mulligans can be performed.")
                                break  # End mulligan phase due to empty deck
                        else:
                            print("You do not have that card in your hand.")
                    except ValueError:
                        print("Please enter a valid card number or 'pass'.")
            
            # After the player finishes or passes, shuffle mulliganed cards back into their deck individually
            while self.buffer_deck:
                card = self.buffer_deck.pop(0)
                self.decks[i].insert(random.randint(0, len(self.decks[i])), card)

            player.hand.sort()
            print(f"Player {i + 1} mulligan phase is over. Hand: {player.hand}")
                


    def play_round(self):

        """Plays a single round of the game."""
        self.reset_round_state()  # Resets scores, passed flags, etc. for the new round
        
        self.game_phase = 0
        
        # Determine the starting player based on the last round's winner
        if self.current_round > 1 and self.last_round_winner is not None:
            starting_player_index = next(i for i, player in enumerate(self.players) if player.player_id == self.last_round_winner)
        else:
            starting_player_index = 0  # Default to Player 1 starting
    
        turn = starting_player_index  # Start with the determined player
        
        print(f"Player {self.players[starting_player_index].player_id} will start Round {self.current_round}.")
        
        
        if self.current_round == 3:  # If it's the final round
            print("Final Round: Automatically summing up cards in hands.")
            for player in self.players:
                # Sum the values of cards in the player's hand
                player.score += sum(player.hand)
                print(f"Player {player.player_id}'s final score: {player.score}")
            
            self.conclude_round()  # Proceed to conclude the round and potentially the game
            return  # Exit the method early since the round is automatically resolved    
        
        while not self.check_round_end_conditions():
            
            current_player = self.players[turn % len(self.players)]
            
            if isinstance(current_player, AIPlayer):
                
                self.print_game_state()
                
                vgs = self.get_visible_game_state(current_player.player_id)
                
                nn_input = self.prepare_nn_input(current_player.player_id)
                nn_input_formatted = np.expand_dims(nn_input, axis=0)
                model_output = model.predict(nn_input_formatted)
                valid_actions_mask = self.get_valid_actions(vgs)
                masked_output = model_output * valid_actions_mask
                selected_action = np.argmax(masked_output)
                
                if selected_action < 16:  # Assuming indices 0-15 correspond to playing cards 1-16
                    # Play the card corresponding to selected_action
                    card_to_play = selected_action + 1  # Adjust index to card number
                    current_player.play_card(card_to_play)
                elif selected_action == 16:  # Assuming index 16 corresponds to passing
                    # Execute a pass action
                    current_player.pass_turn()
            
                turn += 1  # Move to the next player's turn
                
            else:   
                self.print_game_state()
                
                #current_player = self.players[turn % len(self.players)]
                if not current_player.passed:
                    
                    valid_input = False  # Flag to track when a valid input is received
                    while not valid_input:
                        print(f"Player {turn % 2 + 1}'s turn. Hand: {current_player.hand}")
                        action = input("Choose a card to play or type 'pass': ").strip()
        
                        if action.lower() == 'pass':
                            current_player.pass_turn()
                            print(f"Player {turn % 2 + 1} has passed.")
                            valid_input = True  # Valid input received; proceed to next player
                        elif action.lower() == 'n':
                            # Print the output of prepare_nn_input
                            nn_input = self.prepare_nn_input(current_player.player_id)
                            print("NN Input:", nn_input)
                        elif action.lower() == 'g':
                            # Print the output of get_visible_game_state
                            game_state = self.get_visible_game_state(current_player.player_id)
                            print("Visible Game State:", game_state)                        
                        else:
                            try:
                                card_to_play = int(action)
                                if current_player.play_card(card_to_play):
                                    print(f"Player {turn % 2 + 1} plays card {card_to_play}.")
                                    print(f"Player {turn % 2 + 1}'s round score is {current_player.score}.")
                                    valid_input = True  # Valid input received; proceed to next player
                                else:
                                    print("Invalid card, or it's not in your hand.")
                            except ValueError:
                                print("Invalid input. Please enter a card number or 'pass'.")
        
                        if len(current_player.hand) == 0:
                            current_player.pass_turn()
                            print(f"Player {turn % 2 + 1} has no cards left and has passed.")
                            valid_input = True  # Automatically pass if no cards are left
        
                turn += 1  # Move to the next player's turn
        
        self.conclude_round()
        # Determine round winner, update game state, etc.


    def reset_round_state(self):
        """Resets round-specific state, including scores and passed flags."""
        # Reset player scores, passed flags, etc. for the new round
        for player in self.players:
            player.score = 0
            player.passed = False
            player.reset_for_new_round()

        # Apply bonus score to Player 1 at the start of the first round
        if self.current_round == 1:
            self.players[0].score += 3  # Assuming Player 1 is at index 0


    def check_round_end_conditions(self):
        """Checks if the round should end based on game conditions."""
        # Implement logic to check if both players have passed or run out of cards
        return all(player.passed for player in self.players)

    def conclude_round(self):
        """Handles the conclusion of a round."""
        # Calculate round scores for each player (this assumes scores are tracked during the round)
        round_scores = {player.player_id: player.score for player in self.players}

        if len(set(round_scores.values())) == 1:  # This checks if all scores are the same (indicating a tie)
            print("The round ends in a tie.")
            for player in self.players:
                player.round_wins += 1
        else:
            # Determine the round winner based on the scores as before
            winner_id, winner_score = max(round_scores.items(), key=lambda x: x[1])
            loser_id, loser_score = min(round_scores.items(), key=lambda x: x[1])
            print(f"Round {self.current_round} Results: Player {winner_id} wins with {winner_score} points. Player {loser_id} has {loser_score} points.")
            self.players[winner_id - 1].round_wins += 1
            self.last_round_winner = winner_id
        
        # Check if the game has reached a win condition
        # Check for a draw
        if all(player.round_wins >= 2 for player in self.players):
            print("The game ends in a draw.")
            self.game_over = True
        else:
            # Existing logic to check for a game winner
            for player in self.players:
                if player.round_wins >= 2:
                    print(f"Player {player.player_id} wins the game!")
                    self.game_over = True
                    break  # Exit the loop once a winner is found
        
        # Increment the current round only once per round conclusion
        if not self.game_over:  # Ensure we only proceed to the next round if the game isn't over
            round_snapshot = [player.cards_played_this_round.copy() for player in self.players]
            self.played_cards_history.append(round_snapshot)
            self.current_round += 1
            
    def check_game_over(self):
        """Check if the game has reached its end condition."""
        for player in self.players:
            if player.round_wins >= 2:  # Assuming 2 wins are needed to win the game
                self.game_over = True
                print(f"Game Over: Player {player.player_id} wins the game!")
                return True
        return False

    
    def print_game_state(self):
        """Prints the current game state, including scores, round wins, and played cards."""
        print(f"\n--- Game State Before Round {self.current_round} Turn ---")
        for player in self.players:
            print(f"Player {player.player_id}:")
            print(f"  Score this round: {player.score}")
            print(f"  Rounds won: {player.round_wins}")
            print(f"  Cards remaining: {len(player.hand)}")
            print(f"  Cards played this round: {player.cards_played_this_round if player.cards_played_this_round else 'None'}")
        print("------------------------------------------------\n")

    def get_visible_game_state(self, ai_player_id):
        # Find AI player's hand and deck
        ai_hand = next(player.hand for player in self.players if player.player_id == ai_player_id)
        # For the remaining deck, we provide the count of cards, maintaining the unknown order
        ai_deck_cards = next(deck for player, deck in zip(self.players, self.decks) if player.player_id == ai_player_id)
        
        own_played_cards = next(player.cards_played for player in self.players if player.player_id == ai_player_id)
        
        # Shuffle a copy of the AI's remaining deck to hide the order
        shuffled_ai_deck = random.sample(ai_deck_cards, k=len(ai_deck_cards))

        state = {
            "board": [player.cards_played_this_round for player in self.players],
            "remaining_cards": [(len(player.hand), len(deck)) for player, deck in zip(self.players, self.decks)],
            "round_score": {player.player_id: player.score for player in self.players},
            "game_score": {player.player_id: player.round_wins for player in self.players},
            "round_number": self.current_round,
            "game_phase": self.game_phase,
            "ai_hand": ai_hand,  # Cards in the AI player's hand
            "ai_deck_cards": shuffled_ai_deck,  # Shuffled list of remaining cards in the AI's deck
            "played_cards_history": self.played_cards_history,  # History of cards played in previous rounds
            "own_played_cards": own_played_cards
        }
        return state
    
    def get_valid_actions(self, visible_game_state):
        """
        Generate a binary mask indicating valid actions based on the current game state.
        The last position in the mask represents the option to pass.
        """
        # Initialize the mask with zeros - 16 for cards, 1 for pass action
        valid_actions_mask = np.zeros(17, dtype=int)  # 16 cards + 1 pass
        valid_actions_mask[-1] = 1
        
        # Mark cards in hand as valid
        for card in visible_game_state["ai_hand"]:
            valid_actions_mask[card - 1] = 1  # Assuming card numbers start from 1

        return valid_actions_mask  
    
    def prepare_nn_input(self, ai_player_id):
        visible_game_state = self.get_visible_game_state(ai_player_id)
    
        # Encode the player's deck with the encoding discussed: (-1, 0, 1)
        encoded_deck = self.encode_deck(visible_game_state["ai_hand"], visible_game_state["ai_deck_cards"], visible_game_state["own_played_cards"])
    
        # Round score for both players
        round_scores = [visible_game_state["round_score"][player_id]/115 for player_id in sorted(visible_game_state["round_score"])]
    
        # Game score for both players
        game_scores = [visible_game_state["game_score"][player_id] for player_id in sorted(visible_game_state["game_score"])]
    
        # Game phase (0 for mulligan, 1 for normal play)
        game_phase = visible_game_state["game_phase"]
    
        # Number of cards left in each player's deck
        remaining_cards = visible_game_state["remaining_cards"]
    
        # Flatten the remaining_cards to include in the input
        remaining_cards_flat = [item/16 for sublist in remaining_cards for item in sublist]
    
        # Combine all inputs into a single list
        nn_input = encoded_deck + round_scores + game_scores + [game_phase] + remaining_cards_flat
    
        return nn_input
    
    
    def encode_deck(self, ai_hand, ai_deck_cards, own_played_cards):
        # Initialize the deck encoding with zeros
        deck_encoding = [0] * 16  # Assuming 16 unique cards
    
        # Mark cards in hand as 1
        for card in ai_hand:
            deck_encoding[card - 1] = 1  # Adjust for card numbering starting at 1
    
        # Mark played cards as -1 using the own_played_cards list
        for card in own_played_cards:
            deck_encoding[card - 1] = -1
    
        # Cards in the deck are implicitly 0, as they're neither in the hand (1) nor played (-1)
    
        return deck_encoding
    
    def determine_winner(self):
        if self.players[0].round_wins == 2 and self.players[1].round_wins == 2:
            return 0
        elif self.players[0].round_wins == 2:
            return 1
        elif self.players[1].round_wins == 2:
            return 2
        else:
            return None
    
    
def random_ai_logic(visible_game_state):
    ai_hand = visible_game_state["ai_hand"]
    # Decide randomly whether to play a card or pass
    if ai_hand and random.choice([True, False]):  # Assuming AI has cards and chooses randomly to play
        return "play", random.choice(ai_hand)  # Randomly select a card to play
    else:
        return "pass", None  # Decide to pass


def choose_action(model_output, valid_actions):
    # Apply action masking by setting probabilities of invalid actions to 0
    masked_output = np.multiply(model_output, valid_actions)
    
    # Ensure there's at least one valid action by checking the sum of the masked output
    if np.sum(masked_output) > 0:
        # Normalize the masked probabilities so they sum to 1
        probabilities = masked_output / np.sum(masked_output)
        action_index = np.random.choice(range(len(probabilities)), p=probabilities)
    else:
        # Fallback: if no valid actions are probable, default to pass
        action_index = len(valid_actions) - 1  # Last index for "pass"
    
    # Interpret the selected action
    if action_index == len(valid_actions) - 1:
        return "pass", None
    else:
        # Convert action index back to card number (if applicable)
        card_to_play = action_index + 1  # Adjust if your card numbering starts at 1
        return "play", card_to_play



class AIPlayer(Player):
    def __init__(self, player_id, model):
        super().__init__(player_id)
        self.model = model  # The neural network model
        

    def decide_turn(self, game_state):
        prepared_input = np.array([self.prepare_nn_input(game_state)])
        predictions = self.model.predict(prepared_input)[0]
        action = np.argmax(predictions)
        # Translate and execute action
        return action
    
    #####################################

    def decide_action(self, game_state):
        # Process the game state to a format suitable for the model
        processed_game_state = self.process_game_state(game_state)
        
        # Get the model's output (probabilities for each action)
        model_output = self.model.predict(processed_game_state[np.newaxis, :])[0]  # Assuming batch size of 1
        
        # Get valid actions mask based on the current game state
        valid_actions = self.get_valid_actions(game_state)
        
        # Use the choose_action utility function to pick a valid action based on model output and valid actions
        action, card = choose_action(model_output, valid_actions)
        
        return action, card




    def process_game_state(self, visible_game_state):
        """
        Transforms the visible game state into a neural network-friendly format.
        """

        # Initialize a binary vector for the hand, where each position indicates whether
        # the AI has a specific card (1) or not (0). Assumes cards are numbered 1 to 16.
        hand_vector = np.zeros(16, dtype=float)
        for card in visible_game_state['ai_hand']:
            hand_vector[card - 1] = 1.0  # Mark the card as present in the hand

        # The number of cards left in the AI's deck
        deck_size = visible_game_state['ai_deck_remaining']

        # Round scores and rounds won for both players
        round_scores = visible_game_state['round_score']
        game_scores = visible_game_state['game_score']

        # Assuming player IDs are 1 and 2, and AI is player 2
        ai_round_score = round_scores[2]
        opponent_round_score = round_scores[1]
        ai_game_score = game_scores[2]
        opponent_game_score = game_scores[1]

        # Combine all features into a single input vector for the neural network
        nn_input = np.concatenate([
            hand_vector,  # Binary vector for the hand
            [deck_size,  # Number of cards left in the deck
             ai_round_score, opponent_round_score,  # Current round scores
             ai_game_score, opponent_game_score]  # Rounds won
        ])

        return nn_input

    
    def interpret_prediction(self, prediction, ai_hand):
        # Convert the model's prediction into a specific action and card to play
        # This might involve selecting the highest probability action, ensuring it's a valid move, etc.
        pass

########################


# =============================================================================
# def build_model(input_shape):
#     model = Sequential([
#         InputLayer(input_shape=(input_shape,)),  # Input layer specifying the shape of the input vector
#         Dense(64, activation='relu'),  # First hidden layer with 64 neurons and ReLU activation
#         Dense(32, activation='relu'),  # Second hidden layer with 32 neurons and ReLU activation
#         Dense(17, activation='softmax')  # Output layer with 17 neurons (16 cards + 1 for pass) and softmax activation
#     ])
#     
#     model.compile(optimizer='adam',  # Using Adam optimizer
#                   loss='categorical_crossentropy',  # Categorical crossentropy as the loss function for multi-class classification
#                   metrics=['accuracy'])  # Tracking accuracy as the performance metric
#     
#     return model
# 
# ##################
# 
# model = build_model(25)
# 
# model.save('my_model.h5')
# 
# =============================================================================



model = load_model('path_to_my_model.h5')

# Initialize players and the game instance
#player1 = Player(1)
player1 = AIPlayer(1, model)
#player2 = Player(2)
player2 = AIPlayer(2, model)

def play_game(player1, player2):

    game = Game(player1, player2)
    
    game.reset()
    
    # Shuffle decks at the start of the game
    game.shuffle_decks()
    
    # Main game loop
    while not game.game_over:
        print(f"Starting Round {game.current_round}:")
        game.reset_round_state()  # Reset round state
        game.draw_hand()  # Draw hands for the round
        
        # Optionally print hands for debugging/visibility
        print("Player 1's hand:", player1.hand)
        print("Player 2's hand:", player2.hand)
        
        game.start_round_mulligan_phase()  # Handle mulligans
        game.play_round()  # Play out the round
        
        # Check if the game has ended after the round
        if game.check_game_over():
            break  # Exit the loop if the game is over
        
        print(f"Round {game.current_round - 1} Ended\n")
        
        # Print the current round score for each player
        print(f"Player 1's score: {player1.score}")
        print(f"Player 2's score: {player2.score}\n")  # Added another \n for spacing after scores
    
        #game.current_round += 1  # Prepare for the next round
    
    print("Game concluded.")
    
    winner = game.determine_winner()
    return winner

def trial(N, player1, player2):
    
    player1_wins = 0
    player2_wins = 0
    draws = 0
    bugs = 0

    for _ in range(N):
        w = play_game(player1, player2)
        if w == 1:
            player1_wins += 1
        elif w == 2:
            player2_wins += 1
        elif w ==0:
            draws += 1
        else:
            bugs +=1
    
    print(f"Player 1 Wins: {player1_wins}")
    print(f"Player 2 Wins: {player2_wins}")
    print(f"Draws: {draws}")
    print(f"Bugs: {bugs}")

#o = play_game(player1, player2)
#trial(200, player1, player2)

class CustomCardGameEnv(gym.Env):
    def __init__(self, game):
        super(CustomCardGameEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(17)  # 16 cards + 1 for pass
        
        # Example for observation space, adjust according to your game state
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.float32)
        
        self.game=game

    def step(self, action):
        # Execute one time step within the environment
        # You should return four values:
        # observation (object), reward (float), done (boolean), info (dict)
        observation, reward, done, info = ..., ..., ..., ...
        return observation, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.game.reset()
        self.game.shuffle_decks()
        self.game.draw_hand()
        
        self.current_player_id = random.choice([1, 2])  # or  self.current_player_id = 1
        
        # Return initial observation
        observation = self.prepare_nn_input(self.current_player_id)
        return observation































