import  numpy as np
import scipy.special
class NeuralNetwork:
    def __init__(self,inputnodes,hiddenodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddenodes
        self.onodes=outputnodes
        self.lrate=learningrate
        # self.wih=np.random.rand(3,3)-.5 #weight between input and hidden layet
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes))  # this will also give random weight but its according to formula (weight =1/sqrt(number of incoming links).
        # pow(self.hnodes, -0.5)=standard deviation
        # (self.hnodes, self.inodes)=size of matrix
        # self.who = np.random.rand(3, 3) - .5 #weight between hidden and output layer
        self.who = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))#this will also give random weight but its according to formula (weight =1/sqrt(number of incoming links).
        self.activation_function = lambda x: scipy.special.expit(x)
        pass


    def train(self,input_list,target_list):
        input_array = np.array(input_list, ndmin=2).T  # here we need transpose matrix
        target_array = np.array(target_list, ndmin=2).T  # here we need transpose matrix
        # print "target array", target_array
        #hidden layer procession
        hidden_input = np.dot(self.wih, input_array)
        hidden_output = self.activation_function(hidden_input)
        #final layer processing
        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        # print "final output array", final_output
        output_error=target_array-final_output

        # print "the outut error is" , output_error
        print ("the accuracy is ", np.sum(output_error))
        hidden_error=np.dot(self.who.T,output_error)

        #now calculating final weight  changes
        delta_weight=self.lrate*np.dot((output_error*final_output*(1-final_output)),np.transpose(hidden_output))
        self.who+=delta_weight
        # print "the delta weight is ",delta_weight
        self.wih+=self.lrate*np.dot((hidden_error*hidden_output*(1-hidden_output)),np.transpose(input_array))
        pass

# THIS is consider as testing functions
    def query(self,input_list):
        input_array=np.array(input_list,ndmin=2).T#here we need transpose matrix
        hidden_input=np.dot(self.wih,input_array)
        hidden_output=self.activation_function(hidden_input)

        final_input=np.dot(self.who,hidden_output)
        final_output=self.activation_function(final_input)
        return final_output
        pass

myneural=NeuralNetwork(inputnodes=3,hiddenodes=3,outputnodes=3,learningrate=0.5)

print ("prev weight wih /n:", myneural.wih)
print
print ("prev weight who /n:" ,myneural.who)
# print myneural.query([-.6,.5,-.8])

input_l=[.2,.5,.3]
output_l=[.8,.2,.5]

for looper in range(0,100):
    myneural.train(input_l,output_l)


print ("new weight wih /n:", myneural.wih)
print ("new weight who /n:" ,myneural.who)
# print "final output array", myneural.final_output