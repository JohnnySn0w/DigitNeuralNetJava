
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
import java.util.Arrays;
import java.util.Map;
import java.util.Scanner; //user input
import java.util.stream.Stream;



// store trained network in json or something
public class nn {
  /*****************
   * general vars  *
   *****************/
  private static Boolean isLoaded = false; //class var to tell other parts if a nn is loaded
  private static int n = 20; //layer 1 length, between 15 and 30
  private static int k = 10; //layer 2 length, 10 for the 10 digits 0-9
  private static int j = 784; //number of input layer inputs
  private static int arrySize = (j+1)*n+(n+1)*k;
  private static double[] neurNet = new double[arrySize];
  private static String nnetFile = "./neuralNet.csv";
  private static String nnTraining = "./mnist_train.csv";
  private static String nnTest = "./mnist_test.csv";
  private static String[] dataset;
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

  /*******
  * MAIN *
  ********/
  public static void main(String[] args) {
    nn.menu();
  }

  /********************
  * Read in a dataset *
  *********************/
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

  /**********************
  * load an existing nn *
  ***********************/
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

  /**********************
  * store current nn *
  ***********************/
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

   /**********************
  *  initialize net  *
  ***********************/
  static void initNet() {
    //set random wieghts n biasses
    // double[] Arrays.stream(dataset[i].split(delimiter)).mapToDouble(Double::parseDouble).toArray();
    for(int i = 0; i < neurNet.length; i++) {
      neurNet[i] = Math.random();
    }
  }

  /*

    MIGHT HELP TO MAKE THE WHOLE THING RECURSE
  */
  /**********************
  *  sim layers  *
  ***********************/
  //done
  static void layerActivation() {
    for(int i = 0; i < nn.n; i++) {
      neuronActivation(inputs, weights, bias);
    }
  }

  /********************************
  *  simulate a neuron on the net  *
  **********************************/
  static double neuronActivation(double[] inputs, double[] weights, double bias) {
    //take the weights and bias array and math it with the inputs
    //a neuron does a func to the
    double sigmoid = 0.0;
    for(int i = 0; i < weights.length; i++) {
      sigmoid += inputs[i] * weights[i];
    }

    // sigmoid = Arrays.stream(inputs)
    // .mapToInt(input, weight -> input * weights)
    // .sum();

    return sigmoid += 1/(1 + Math.exp(-sigmoid-bias)); // sigmoidd
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
        nn.parseDataset(nnTraining, 60000);
        System.out.println("training");
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
          System.out.println("5 runs");
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