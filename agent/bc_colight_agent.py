from . import RLAgent, BaseAgent
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.engine.topology import Layer


class RepeatVector3D(Layer):
    """
    expand axis=1, then tile times on axis=1
    """
    def __init__(self,times,**kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.times, input_shape[1],input_shape[2])

    def call(self, inputs):
        #[batch,agent,dim]->[batch,1,agent,dim]
        #[batch,1,agent,dim]->[batch,agent,agent,dim]
        return K.tile(K.expand_dims(inputs,1),[1,self.times,1,1])


    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BCCoLightAgent(BaseAgent):
    def __init__(self, action_space, ob_generator, world, traffic_env_conf, graph_setting, args):
        super().__init__(action_space)

        self.ob_generators = ob_generator # a list of ob_generators for each intersection ordered by its int_id
        self.ob_length = ob_generator[0][1].ob_length # the observation length for each intersection

        self.graph_setting = graph_setting
        self.neighbor_id = self.graph_setting["NEIGHBOR_ID"] # neighbor node of node
        self.degree_num = self.graph_setting["NODE_DEGREE_NODE"] # degree of each intersection
        self.direction_dic = {"0":[1,0,0,0],"1":[0,1,0,0],"2":[0,0,1,0],"3":[0,0,0,1]} # TODO: should refine it

        self.dic_traffic_env_conf = traffic_env_conf
        self.num_agents=self.dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_edges = self.dic_traffic_env_conf["NUM_ROADS"]

        self.vehicle_max = args.vehicle_max
        self.mask_type = args.mask_type

        self.batch_size = args.batch_size #32
        self.world = world
        self.world.subscribe("pressure")
        self.world.subscribe("lane_count")
        self.world.subscribe("lane_waiting_count")

        self._placeholder_init()
        self._build_bc_net()
        self.bc_params = tf.get_collection("bc_net")
        t_config = tf.ConfigProto()
        t_config.gpu_options.allow_growth = True
        self.algo_saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=None)
        self.sess = tf.Session(config=t_config)
        self.sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log/', tf.get_default_graph())

    def _placeholder_init(self):
        self.node_state_length = 0
        self.input_node_state=tf.placeholder(dtype=tf.float32, shape=[None,self.num_agents,self.ob_length],name='input_state') # concat [lane_num_vehicle]
        self.input_node_phase=tf.placeholder(dtype=tf.int32, shape=[None,self.num_agents],name='input_phase')
        self.input_neighbor_id = tf.placeholder(dtype=tf.int32, shape=[self.num_agents,self.graph_setting["NEIGHBOR_NUM"]],name="neighbor_id") #neighbor node of node
        self.input_node_degree_mask = tf.placeholder(dtype=tf.int32,shape=[self.num_agents], name="node_degree_mask") #not all nodes has degree 4
        self.input_actions = tf.placeholder(dtype=tf.int32, shape=[None,self.num_agents], name="input_actions") #batch ,agents
        # self.input_target_value = tf.placeholder(dtype=tf.float32, shape=[None,self.num_agents,],name="input_target_value")
        one_hot_phase = tf.one_hot(self.input_node_phase, len(self.world.intersections[0].phases)) # batch, agents, num_phases
        self.concat_input = tf.concat([self.input_node_state,one_hot_phase],axis=-1,name="concate_input_obs")
        #actions one-hot
        self.input_actions_one_hot = tf.one_hot(self.input_actions, self.action_space.n)
        # the adjacent node of node
        #self.neighbor_node_one_hot = tf.one_hot(self.input_neighbor_id,self.num_agents) # agent, neighbor, agents  
        neighbor_node_tmp = tf.range(0,self.num_agents)
        neighbor_node_tmp = tf.expand_dims(neighbor_node_tmp,axis=-1)
        expanded_neighbor_id = tf.concat([neighbor_node_tmp,self.input_neighbor_id],axis=-1) #[agents ,num_neighbor+1] include node itself
        neighbor_node_one_hot = tf.one_hot(expanded_neighbor_id,self.num_agents)
        neighbor_node_one_hot = tf.expand_dims(neighbor_node_one_hot,axis=0)
        self.neighbor_node_one_hot = tf.tile(neighbor_node_one_hot,[tf.shape(self.input_node_state)[0],1,1,1])
        # process the mask
        degree_mask = self.input_node_degree_mask + 1 
        degree_mask = tf.sequence_mask(degree_mask,self.graph_setting["NEIGHBOR_NUM"]+1) # agetns, neighbor_num+1
        # degree_mask = tf.expand_dims(degree_mask,axis=-1) 
        self.degree_mask = tf.cast(degree_mask,tf.float32,name="processed_degree") #agents, neighbor_num
        self.actions_expert = tf.placeholder(tf.int32, shape=[None, self.num_agents], name='actions_expert')

    def _build_model(self):
        """
        layer definition
        """
        """
        #[#agents,batch,feature_dim],[#agents,batch,neighbors,agents],[batch,1,neighbors]
        ->[#agentsxbatch,feature_dim],[#agentsxbatch,neighbors,agents],[batch,1,neighbors]
        """
        feature=self.MLP(self.input_node_state,self.graph_setting["NODE_EMB_DIM"]) #feature:[batch,agents,feature_dim]
        
        att_record_all_layers=list()
        for i in range(self.graph_setting["N_LAYERS"]):
            if i==0:
                h,att_record=self._MultiHeadsAttModel(
                    feature,
                    self.neighbor_node_one_hot,
                    l=self.graph_setting["NEIGHBOR_NUM"],
                    d=self.graph_setting["INPUT_DIM"][i],
                    dv=self.graph_setting["NODE_LAYER_DIMS_EACH_HEAD"][i],
                    dout=self.graph_setting["OUTPUT_DIM"][i],
                    nv=self.graph_setting["NUM_HEADS"][i],
                    suffix=i
                    )
            else:
                h,att_record=self._MultiHeadsAttModel(
                    h,
                    self.neighbor_node_one_hot,
                    l=self.graph_setting["NEIGHBOR_NUM"],
                    d=self.graph_setting["INPUT_DIM"][i],
                    dv=self.graph_setting["NODE_LAYER_DIMS_EACH_HEAD"][i],
                    dout=self.graph_setting["OUTPUT_DIM"][i],
                    nv=self.graph_setting["NUM_HEADS"][i],
                    suffix=i
                    )
            att_record_all_layers.append(att_record)
        return h,att_record_all_layers

    def _build_bc_net(self):
        with tf.variable_scope("bc_net", reuse=tf.AUTO_REUSE):
            value_output, att_record_eval = self._build_model()
            value_output = tf.reshape(value_output, shape = [-1, 128])
            acts = Dense(self.action_space.n, kernel_initializer='random_normal', name='obs_embedding')(value_output)
            self.act_probs = Dense(self.action_space.n, activation='softmax', kernel_initializer='random_normal', name='act_probs')(acts)
            # self.act_probs = tf.layers.dense(inputs=layer_3, units=self.action_space.n, activation=tf.nn.softmax)
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])
            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            actions_expert = tf.reshape(self.actions_expert, shape = [-1])
            actions_vec = tf.one_hot(actions_expert, depth=self.act_probs.shape[1], dtype=tf.float32)
            loss = tf.reduce_sum(actions_vec * tf.log(tf.clip_by_value(self.act_probs, 1e-10, 1.0)), 1)
            loss = - tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)
            self.loss = loss

    def _MultiHeadsAttModel(self,In_agent,In_neighbor,l=5, d=128, dv=16, dout=128, nv = 8,suffix=-1):
        """
        input:
            In_agent [bacth,agent,128]
            In_neighbor [agent, neighbor_num]
            l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
            d: dimension of agent's embedding
            dv: dimension of each head
            dout: dimension of output
            nv: number of head (multi-head attention)
        output:
            -hidden state: [batch,agent,32]
            -attention: [batch,agent,neighbor]
        """

        """agent repr"""
        # print("In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv", In_agent.shape,In_neighbor.shape,l, d, dv, dout, nv)
        #[batch,agent,dim]->[batch,agent,1,dim]
        agent_repr=Reshape((self.num_agents,1,d))(In_agent)

        """neighbor repr"""
        #[batch,agent,dim]->(reshape)[batch,1,agent,dim]->(tile)[batch,agent,agent,dim]
        neighbor_repr=RepeatVector3D(self.num_agents)(In_agent)
        # print("neighbor_repr.shape", neighbor_repr.shape)
        #[batch,agent,neighbor,agent]x[batch,agent,agent,dim]->[batch,agent,neighbor,dim]
        neighbor_repr=Lambda(lambda x:K.batch_dot(x[0],x[1]))([In_neighbor,neighbor_repr])
        # print("neighbor_repr.shape", neighbor_repr.shape)
        
        """attention computation"""
        #multi-head
        #[batch,agent,1,dim]->[batch,agent,1,dv*nv]
        agent_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='agent_repr_%d'%suffix)(agent_repr)
        #[batch,agent,1,dv,nv]->[batch,agent,nv,1,dv]
        agent_repr_head=Reshape((self.num_agents,1,dv,nv))(agent_repr_head)
        agent_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(agent_repr_head)

        neighbor_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_repr_%d'%suffix)(neighbor_repr)
        #[batch,agent,neighbor,dv,nv]->[batch,agent,nv,neighbor,dv]
        # print("DEBUG",neighbor_repr_head.shape)
        # print("self.num_agents,self.num_neighbors,dv,nv", self.num_agents,self.graph_setting["NEIGHBOR_NUM"],dv,nv)
        neighbor_repr_head=Reshape((self.num_agents,self.graph_setting["NEIGHBOR_NUM"]+1,dv,nv))(neighbor_repr_head)
        neighbor_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_repr_head)
        
        #should mask
        tmp_mask = tf.expand_dims(self.degree_mask,axis=1) # [agents,neighbor] --->  [agents,1,neighbor]
        tmp_mask = tf.expand_dims(tmp_mask,axis=-2) # [agents,1,neighbor] --->  [agents,1,1,neighbor]
        tmp_mask = tf.tile(tmp_mask,[1,nv,1,1]) # [agents,1,1,neighbor] --->  [agents,nv,1,neighbor]

        if self.mask_type==1:
            att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head]) #[batch,agents,nv,1,neighbor]
            att = att * tmp_mask #[batch,agents,nv,1,neighbor]
            tmp_att = att
            att = att / tf.expand_dims(tf.reduce_sum(tmp_att,axis=-1),axis=-1)
            # print("att.shape:",att.shape)
        else:
            #[batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
            att=Lambda(lambda x:K.softmax(K.batch_dot(x[0],x[1],axes=[4,4])))([agent_repr_head,neighbor_repr_head]) #[batch,agents,nv,1,neighbor]
            att = att * tmp_mask #[batch,agents,nv,1,neighbor]
            # print("att.shape:",att.shape)
        att_record=Reshape((self.num_agents,nv,self.graph_setting["NEIGHBOR_NUM"]+1))(att) #[batch,agent,nv,1,neighbor]->[batch,agent,nv,neighbor]
        


        #self embedding again
        neighbor_hidden_repr_head=Dense(dv*nv,activation='relu',kernel_initializer='random_normal',name='neighbor_hidden_repr_%d'%suffix)(neighbor_repr)
        neighbor_hidden_repr_head=Reshape((self.num_agents,self.graph_setting["NEIGHBOR_NUM"]+1,dv,nv))(neighbor_hidden_repr_head) #[batch,agents,neighbor,dv,nv]
        neighbor_hidden_repr_head=Lambda(lambda x:K.permute_dimensions(x,(0,1,4,2,3)))(neighbor_hidden_repr_head) #[batch,agents,nv,neighbor,dv]
        out=Lambda(lambda x:K.mean(K.batch_dot(x[0],x[1]),axis=2))([att,neighbor_hidden_repr_head]) # [batch,agents,nv,1,neighbor]*[batch,agents,nv,neighbor,dv]--->[batch,agents,nv,1,dv]--->[batch,agents,1,dv]
        out=Reshape((self.num_agents,dv))(out) #[batch, agents,dv]
        out = Dense(dout, activation = "relu",kernel_initializer='random_normal',name='MLP_after_relation_%d'%suffix)(out)
        # print("out-shape:", out.shape)
        return out,att_record
    
    def MLP(self,In_0,layers=[128,128]):
        """
        Currently, the MLP layer 
        -input: [batch,#agents,feature_dim]
        -outpout: [batch,#agents,128]
        """
        # In_0 = Input(shape=[self.num_agents,self.len_feature])
        for layer_index,layer_size in enumerate(layers):
            if layer_index==0:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(In_0)
            else:
                h = Dense(layer_size, activation='relu',kernel_initializer='random_normal',name='Dense_embed_%d'%layer_index)(h)

        return h

    def get_actions(self, obs):
        """
        phase : [agents]
        obs : [agents, ob_length]
        edge_obs : [agents, edge_ob_length]
        return: [batch,agents]
        we should expand the input first to adapt the [batch,] standard
        """
        ob = obs[:, :-1]
        phase = obs[:, -1]
        e_ob = ob[np.newaxis,:]
        e_phase = np.array(phase)
        e_phase = e_phase[np.newaxis,:]

        my_feed_dict = {
            self.input_node_state: e_ob,
            self.input_node_phase: e_phase,
            self.input_neighbor_id: self.neighbor_id,
            self.input_node_degree_mask: self.degree_num
        }
        actions = self.sess.run([self.act_deterministic], feed_dict=my_feed_dict)[0]
        return actions.tolist()
        
    def sample(self):
        # return np.random.randint(0,self.action_space.n,s_size)
        return self.action_space.sample()

    def get_rewards(self):
        return [0 for i in self.world.intersections]

    def get_obs(self):
        """
        return: obersavtion of node, observation of edge
        """
        x_obs = [] # num_agents * lane_nums, 
        for i in range(len(self.ob_generators)):
            node_id_str = self.graph_setting["ID2INTER_MAPPING"][i]
            node_dict = self.world.id2intersection[node_id_str]
            phase = node_dict.current_phase
            obs = (self.ob_generators[i][1].generate())/self.vehicle_max
            obs = np.append(obs, phase)
            x_obs.append(obs)
        # construct edge infomation
        x_obs = np.array(x_obs)
        return x_obs

    def reset(self):
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def update_policy(self, obs, actions):
        # print(obs.shape)
        obs = obs.reshape((self.num_agents * self.batch_size, -1))
        ob = obs[:, :-1]
        phase = obs[:, -1]
        e_ob = ob.reshape((self.batch_size, self.num_agents, -1))
        e_phase = phase.reshape((self.batch_size, self.num_agents))

        my_feed_dict = {
            self.input_node_state: e_ob,
            self.input_node_phase: e_phase,
            self.input_neighbor_id: self.neighbor_id,
            self.input_node_degree_mask: self.degree_num,
            self.actions_expert: actions
        }
        loss, _ = self.sess.run([self.loss,self.train_op],feed_dict=my_feed_dict)
        return loss

    def load_model(self, dir="model/bc", model_id=None):
        if model_id is None:
            self.algo_saver.restore(self.sess, tf.train.latest_checkpoint(dir))
        self.algo_saver.restore(self.sess, dir+"-"+str(model_id))

    def save_model(self, step, dir="model/bc"):
        self.algo_saver.save(self.sess, dir, global_step=step)