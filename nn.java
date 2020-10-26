
/**
 * nn, a neural net in Java, specifically for recognizing handwritten digits
 * Name: Michael Mahan
 * Student ID: 102-36-293
 * Wednesday, October 14, 2020
 * Assignment 2
 * Class: CSC-475
 * Professor: Dr. Mike O'Neal
 * Initial temperature: 63 Farenhiet in Ruston, LA
 * Initial start time: 8:58 PM
 * Initial start mood: Mildly Trepidatious
 * End temperature
 * End time:
 * End mood:
 * Assessment:
 */

import java.io.FileReader; // self-explanatory
import java.io.FileWriter;
import java.io.BufferedReader; // buffering file input
import java.io.IOException; // error handling
// import java.util.ArrayList;
// import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner; //user input



// store trained network in json or something
public class nn {
  /*****************
   * general vars  *
   *****************/
  private static Boolean isLoaded = false; //class var to tell other parts if a nn is loaded
  // private static int j = 784; //number of input layer inputs
  // private static int n = 20; //layer 1 length, between 15 and 30
  // private static int k = 10; //layer 2 length, 10 for the 10 digits 0-9
  private static int[] layers = {784, 20, 10};
  private static int arrySize = (layers[0]+1)*layers[1]+(layers[1]+1)*layers[2];
  private static double[] neurNet = new double[arrySize];
  private static String nnetFile = "./neuralNet.csv";
  private static String nnTraining = "./mnist_train.csv";
  private static String nnTest = "./mnist_test.csv";
  private static String[] dataset;
  private static double[][] layerOutputs = new double[layers.length-1][]; //arrayList allows for just enough nodes
  /*
    this is a bit of a hack I borrow from 3d graphics, setting what would 
    be a multidimensional array into a smaller array for better access times.
  */
  /*
  the input layer just passes the input value, then each neuron on the next layer, 
  takes all 784 inputs as an input, with a weight per input(so the weights are per incoming neuron, 
  but assigned per individual next neuron), which means the shape should be like, 784 weights on the first layer,plus a bias at the end, per first layer node
  so then to abstract this to a 1d array, 785 elements per 1st layer neuron, times n, which should be 20 default.
  785*20 indices in is where layer 2 starts, which should be 21 per neuron, 21*10. The total array size should be (784+1)*20+(20+1)*10, which is 15910
  so if I access index 15721, that's the 2nd layer, bias for the first neuron
  */
  // why even bother with a 1d array, seems complex? Speed.

  //THE FIRST ELEMENT OF EACH ROW IS THE ANSWER, DONT COMPUTE IT

  /*******
  * MAIN *
  ********/
  public static void main(String[] args) {
    nn.menu();
  }

  /*******************************
  * Store and Load datasets/nets *
  ********************************/
  //done
  static void parseDataset(String datasetURI, int datasetSize) {
    String line = ""; //instantiation for later, this will hold each full line of pixel values
    dataset = new String[datasetSize]; // each row is an image
    try(BufferedReader buffer = new BufferedReader(new FileReader(datasetURI))) { //try reading in the file
      int i = 0;
      while((line = buffer.readLine()) != null) {
        dataset[i] = line; // each line is 1/60000 of the set
        i++;
      }
    }
    catch (IOException e) {
      e.printStackTrace(); //debugging
    }
    // System.out.println(nn.dataset[0]);
  }

  //done(excpet maybe not)
  static boolean loadNet() {
    String line = "";
    String delimiter = ", ";
    try {
      FileReader nnFile = new FileReader(nnetFile);
      BufferedReader nnBuffer = new BufferedReader(nnFile);
      line = nnBuffer.readLine();
      neurNet = Arrays.stream(line.split(delimiter)).mapToDouble(Double::parseDouble).toArray();
      System.out.println(neurNet.length);
      nnBuffer.close();
    }
    catch (IOException e) {
      System.out.println("error with reading file");
      return false;
    }
    catch (NullPointerException e) {
      System.out.println("No dataset detected in current directory");
      return false;
    }
    return true;
  }

  // done
  static boolean storeNet() {
    try {
      FileWriter netFile = new FileWriter(nnetFile, false);
      String stringifiedNet = Arrays.toString(nn.neurNet);
      stringifiedNet = stringifiedNet.replaceAll("^.|.$", "");
      netFile.write(stringifiedNet);
      netFile.close();
    }
    catch (IOException e) {
      e.printStackTrace(); //debugging
      return false;
    }
    return true;
  }

  static void initNet() {
    //set random wieghts n biasses
    // double[] Arrays.stream(dataset[i].split(delimiter)).mapToDouble(Double::parseDouble).toArray();
    for(int i = 0; i < neurNet.length; i++) {
      neurNet[i] = Math.random();
    }
    isLoaded = true;
  }

  /**************
   * batches area *
   **************/

  static double[][] generateBatches(){
    //chop up the dataset using random and make 6 subsets of 10k each or whatever
    //after finishing the initial bits, parameterize it
  }

  // static double[] convertRowToDouble(String row){
    
  //   return ;
  // }

   /**********************
  *  NN running functions  *
  ***********************/
  static void runNet(){
    //init the layer lengths in jagged array
    for(int layerIndex = 1; layerIndex < layers.length; layerIndex++){
      layerOutputs[layerIndex-1] = new double[layers[layerIndex]]; //layerOutputs is layers-1 since we don't need to store l0
      layerActivation(layerIndex);
    }
  }
 
  static void layerActivation(int layerIndex) {
    // split out the current layer from the network
    //we only have 2 layers, so layer 1 and layer 2
    //layer 0 is just the base inputs
    int srcPos, destPos, length;
    double[] inputs;
    //assume layerindex is never less than 1 bc 0 is just inputs(no calc needed)
    //works for any nnumber of layers(?)
    if(layerIndex > 1){ //TODO: could simplify this logic if java does implicit bool stuff
      srcPos = (layers[layerIndex-1]*(layers[layerIndex-2]+1)); //layer 1 nodes * weights + bias(elements per node) for layer 0 gives index of last element
      destPos = 0;
      length = layers[layerIndex]*(layers[layerIndex-1]+1);
      int ending = length+srcPos-1;
      System.out.println("start:"+srcPos+"end:"+ending+"size:"+length+"\n");//should be 15700 and 15909
    } else {
      srcPos = 0; //layer 1 starts at 0
      destPos = 0;
      length = (layers[1]*(layers[0]+1)); //last index before layer 2 starts
      int ending = length+srcPos-1;
      System.out.println("start:"+srcPos+"end:"+ending+"size:"+length+"\n"); //should be 0 and 15699
      inputs = convertRowToDouble();
    }
    double[] layerWeightsNBiases = new double[length];  //this should be an entire set of weights+a bias for a layer
    System.arraycopy(neurNet, srcPos, layerWeightsNBiases, destPos, length); 
    for(int nodeIndex = 0; nodeIndex < nn.layers[1]; nodeIndex++) {
      double[] inputs = convertRowToDouble(row);
      neuronActivation(inputs, weights, bias, layerIndex, nodeIndex);
    }
  }

  //done??
  static void neuronActivation(double[] inputs, double[] weights, double bias, int layerIndex, int nodeIndex) {
    //take the weights and bias array and math it with the inputs
    //a neuron does a func to the
    double preSigmoid = 0.0;
    for(int i = 0; i < weights.length; i++) {
      preSigmoid += inputs[i] * weights[i];
    }
    // sigmoid = Arrays.stream(inputs)
    // .mapToInt(input, weight -> input * weights)
    // .sum();
    double sigmoid = 1/(1 + Math.exp((-preSigmoid)-bias));
    nn.layerOutputs[layerIndex-1][nodeIndex] = sigmoid; //store neuron's output to layerOutputs
    //its layerIndex-1 because of the lack of l0 in the layerOutputs, so everything is offset
  }


  /***************
  * Menu Options *
  ****************/
  static void menu() {
    String extras = isLoaded ? ("[3] Display network accuracy on TRAINING data\n" +
    "[4] Display network accuracy on TESTING data\n" +
    "[5] Save the network state to file\n") : "";

    System.out.println("Weclome to the nn program, here we train networks to identify handwritten digits.\n"+
    "Please select from the Following:\n" +
    "[1] Train the network\n" +
    "[2] Load a pre-trained network\n" +
    extras +
    "[0] Exit Program");

    // create an object of Scanner to read input
    Scanner input = new Scanner(System.in);
    // take input from the user
    String command = input.next();
    switch (command) {
      case "0":
        System.exit(0);
        break;
      case "1":
        //make a net, train it on the training set
        initNet();
        System.out.println("training");
        runNet();
        break;
      case "2":
        System.out.println("loading");
        isLoaded = nn.loadNet();
        break;
      case "3": //test against training data
        if(isLoaded) {
          nn.parseDataset(nnTraining, 60000);
          System.out.println("3 runs");
        }
        break;
      case "4": //test against testing data
        if(isLoaded) {
          nn.parseDataset(nnTest, 60000);
          System.out.println("4 runs");
        }
        break;
      case "5": //save net
        if(isLoaded) {
          nn.storeNet();
        }
        break;
      default:
        nn.menu();
        break;
    }
    nn.menu(); //call the menu again so it refreshes
    input.close(); //close the scanner(this seems to be good practice/necessary)
  }

}