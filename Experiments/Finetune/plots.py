import matplotlib.pyplot as plt
import numpy as np

# Data for RoBERTa_Large
steps_roberta_wp = np.array([100, 200, 300, 400, 500, 600, 700])
accuracy_roberta_wp = np.array([0.302, 0.458, 0.604, 0.572, 0.437, 0.645, 0.625])

steps_roberta_sp = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
accuracy_roberta_sp = np.array([0.375, 0.558, 0.666, 0.675, 0.700, 0.783, 0.758, 0.775, 0.758, 0.766])

# Data for BERT
steps_bert_wp = np.array([100, 200, 300, 400, 500, 600, 700])
accuracy_bert_wp = np.array([0.406, 0.593, 0.531, 0.593, 0.604, 0.593, 0.614])

steps_bert_sp = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
accuracy_bert_sp = np.array([0.541, 0.591, 0.583, 0.633, 0.675, 0.641, 0.708, 0.733, 0.700, 0.708])

# Create the figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12,6))

# Plot lines for Word Puzzles
axs[0].plot(steps_roberta_wp, accuracy_roberta_wp, 'ro-', label='RoBERTa_Large')
axs[0].plot(steps_bert_wp, accuracy_bert_wp, 'bo-', label='BERT')

# Add title and labels to axes for Word Puzzles
axs[0].set_title('Word Puzzles')
axs[0].set_xlabel('Training Steps')
axs[0].set_ylabel('Accuracy')

# Show legend to identify each line for Word Puzzles
axs[0].legend()

# Plot lines for Sentence Puzzles
axs[1].plot(steps_roberta_sp, accuracy_roberta_sp, 'ro-', label='RoBERTa_Large')
axs[1].plot(steps_bert_sp, accuracy_bert_sp, 'bo-', label='BERT')

# Add title and labels to axes for Sentence Puzzles
axs[1].set_title('Sentence Puzzles')
axs[1].set_xlabel('Training Steps')
axs[1].set_ylabel('Accuracy')

# Show legend to identify each line for Sentence Puzzles
axs[1].legend()

# Display the grid
axs[0].grid(True)
axs[1].grid(True)

# Adjust the space between plots
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('finetuned_combined.png')

# Show the figure
plt.show()
