import numpy as np
import networkx as nx
import tensorflow as tf

# Các thông số đầu vào
num_episodes = 5000
learning_rate = 0.1
discount_factor = 1
exploration_rate = 0.9
c_start_ep_epsilon_decay = 1
c_end_ep_epsilon_decay = num_episodes
v_epsilon_decay = exploration_rate / (c_end_ep_epsilon_decay - c_start_ep_epsilon_decay)

# Đọc dữ liệu mạng vật lý và mạng SFC
PHY = nx.read_gml("./data_import/data_PHY/giul39.gml")
SFC = nx.read_gml("./data_import/data_SFC/SFC_graph.gml")

# Lớp môi trường cho bài toán SFC Mapping
class SFCMappingEnvironment:
    def __init__(self):
        # Thiết lập mạng vật lý
        self.PHY_nodes = list(PHY.nodes())
        self.PHY_weights_node = [PHY.nodes[node]['weight'] for node in self.PHY_nodes]
        self.PHY_array = nx.adjacency_matrix(PHY).toarray()

        # Thiết lập mạng SFC
        self.SFC_nodes = list(SFC.nodes())
        self.SFC_weights_node = [SFC.nodes[node]['weight'] for node in self.SFC_nodes]
        self.SFC_array = nx.adjacency_matrix(SFC).toarray()

        # Thiết lập không gian trạng thái và hành động
        self.state_space = list(range(len(self.SFC_nodes)))
        self.action_space = list(range(len(self.PHY_nodes)))

    # Phương thức cập nhật trọng số của các node PHY sau khi mapping
    def update_PHY_node_weights(self, mapping_pairs):
        for node_SFC, node_PHY in mapping_pairs:
            self.PHY_weights_node[node_PHY] -= self.SFC_weights_node[node_SFC]

    # ... (các phương thức còn lại của lớp môi trường)

# Xây dựng mô hình Deep Q-Network (DQN) sử dụng TensorFlow
class DQN:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        # Xây dựng mô hình mạng neural
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.state_space_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_space_size)
        ])

        # Sử dụng thuật toán tối ưu hóa Adam để cập nhật trọng số
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Dự đoán hành động dựa trên trạng thái hiện tại
    def predict_action(self, state):
        return np.argmax(self.model.predict(state[np.newaxis, :])[0])

    # Huấn luyện mạng neural
    def train(self, states, actions, rewards, next_states, dones):
        next_state_values = np.max(self.model.predict(next_states), axis=1)
        target_values = rewards + discount_factor * next_state_values * (1 - dones)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            selected_actions = tf.one_hot(actions, self.action_space_size)
            selected_q_values = tf.reduce_sum(tf.multiply(q_values, selected_actions), axis=1)
            loss = tf.reduce_mean(tf.square(selected_q_values - target_values))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

# Chương trình chính
env = SFCMappingEnvironment()
dqn = DQN(state_space_size=len(env.state_space), action_space_size=len(env.action_space))

# Hàm chọn hành động dựa trên mô hình DQN và epsilon-greedy exploration
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(env.action_space)
    else:
        return dqn.predict_action(state)

# Quá trình training với thuật toán DQL
for episode in range(num_episodes):
    state = np.random.choice(env.state_space)
    epsilon = max(0.01, exploration_rate - (exploration_rate / num_episodes) * episode)
    total_reward = 0

    while True:
        action = choose_action(np.array([state]), epsilon)
        next_state = env.state_space[np.random.choice(len(env.state_space))]
        reward = 1000 - (env.PHY_weights_node[action] - env.SFC_weights_node[state])
        done = False if len(env.state_space) > 1 else True

        # Cập nhật trọng số của node PHY sau khi mapping
        if len(env.state_space) > 1:
            num_hop, env.PHY_array, total_weight_edge = dijkstra(env.PHY_array, int(action), env.state_space[next_state], env.SFC_array[state][env.state_space[next_state]])
            reward -= 5 * num_hop + 3 * (total_weight_edge - (num_hop - 1) * env.SFC_array[state][env.state_space[next_state]])

        total_reward += reward

        dqn.train(np.array([state]), action, reward, np.array([next_state]), done)

        if done:
            break

        state = next_state

# Sau khi training, chọn hành động tốt nhất cho mỗi trạng thái
best_actions = [dqn.predict_action(np.array([state])) for state in env.state_space]

# Ánh xạ SFC xuống PHY bằng cách chọn hành động tốt nhất cho mỗi trạng thái
mapping_pairs = [(state, action) for state, action in zip(env.state_space, best_actions)]
print("Mapping pairs:", mapping_pairs)

# Cập nhật trọng số của node PHY sau khi ánh xạ SFC
env.update_PHY_node_weights(mapping_pairs)
