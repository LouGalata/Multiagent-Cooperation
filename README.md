**Multiagent Cooperation**

*Environment*
1. Multiagent particle environment: https://github.com/openai/multiagent-particle-envs
2. BATTLE (https://github.com/PKU-AI-Edge/DGN): in progress

*Experiments*
The configuring parameters are located on the top of each execution file. 
1. train-iql.py: Independent q-learning with VDN mixing strategy
2. train-iql-prior.py: Independent q-learning with VDN mixing strategy and prioritized experience replay
3. train-gcn.py: Graph Convolutional Network with VDN 
4. train-gat.py: Graph Attentional Network with VDN 
5. train-gat-ind.py: Graph Attentional Network with VDN without shared weights 
6. train-gat-entr.py: Graph Attentional Network with VDN and maximum entropy regularization
7. train-rnn-ind.py:  Independent q-learning with recurrence and VDN mixing strategy 
8. train-dueling-dqn.py: Independent Dueling DQN with VDN mixing strategy
9. train-maddpg.py: MADDPG

*Buffers*
1. replay_buffer.py: Save state, action, adjacency_matrix, next_action, reward, done
2. replay_buffer_entr.py: Save state, action, adjacency_matrix, next_action, reward, done, entropy
3. replay_buffer_iql.py: Save state, action, next_action, reward, done (without GNN)
4. prioritized_replay_buffer.py

Statistics and best models are saved under the **results** folder. In *plotting.py*, we are plotting the loss per episode and the evaluation reward per episode.

