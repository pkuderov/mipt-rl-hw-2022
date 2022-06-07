class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output

        #print("DEBUG argmax policy")
        #print("shape:", self.critic.qa_values(observation).shape)
        #self.critic.qa_values(observation)
        action = self.critic.qa_values(observation).argmax()
        #temp = self.critic.qa_values(observation)
        #action = temp.argmax()

        #print("DEUBG argmax")
        #print(temp)
        #print(action)

        return action.squeeze()
