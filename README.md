**Multiagent Cooperation**

*Environment*
1. Multiagent particle environment: https://github.com/openai/multiagent-particle-envs 

*Experiments*
The configuring parameters are located on the top of each execution file. 
1. train-iql.py: IQL with VDN mixing strategy
3. train-gcn.py: IQL enhanced with Graph Convolutional Network with VDN 
4. train-gat.py: IQL enhanced with Graph Attentional Network with VDN 
5. train-gat-ind.py: IQL enhanced with Graph Attentional Network with VDN without shared weights 
7. train-rnn-ind.py:  IQL with recurrence and VDN mixing strategy 
8. train-dueling-dqn.py: Independent Dueling DQN with VDN mixing strategy
9. train-maddpg.py: MADDPG
10. train-centr-maddpg: MADDPG with one centralized critic

*Buffers*
1. replay_buffer.py: Save state, action, adjacency_matrix, next_action, reward, done
3. replay_buffer_iql.py: Save state, action, next_action, reward, done (without GNN)
4. prioritized_replay_buffer.py

Statistics and best models are saved under the **results** folder. In *plotting.py*, we are plotting the loss per episode and the evaluation reward per episode.

