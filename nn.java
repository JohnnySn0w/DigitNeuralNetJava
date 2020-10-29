
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
 * End temperature:
 * End time:
 * End mood: 
 * Assessment:
 */

import java.io.FileReader; // self-explanatory
import java.io.FileWriter;
import java.io.BufferedReader; // buffering file input
import java.io.IOException; // error handling
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
  private static int[] layers = {784, 15, 10};
  private static int arrySize = (layers[0]+1)*layers[1]+(layers[1]+1)*layers[2];
  private static double[] neurNet = new double[arrySize]; //weights and bias per node, sequentially
  private static double[] gradients = new double[arrySize];//weight and bias gradients per round, for simplicity, to be stored 
  private static String nnetFile = "./neuralNet.csv";
  private static String nnTraining = "./mnist_train.csv";
  private static String nnTest = "./mnist_test.csv";
  private static ArrayList<String> dataset;
  private static double[][] layerOutputs = new double[layers.length-1][]; //jagged array for reduced footprint
  private static int batchSize = 100; //how big batches should be
  private static double learningRate = 3.0;
  private static int[] correct = new int[10];//correct per output
  private static int[] total = new int[10];//total per number
  private static int[] oneHotVector = new int[10]; // sexy/10
  private static double[][] gradientsPerBatch = new double[batchSize][arrySize]; //first index
  private static int epochsToRun = 6;
  private static boolean testing = false;
  /*
    Never do this again
  */
  /*
  the input layer just passes the input value, then each neuron on the next layer, 
  takes all 784 inputs as an input, with a weight per input(so the weights are per incoming neuron, 
  but assigned per individual next neuron), which means the shape should be like, 784 weights on the first layer,plus a bias at the end, per first layer node
  so then to abstract this to a 1d array, 785 elements per 1st layer neuron, times n, which should be 20 default.
  785*20 indices in is where layer 2 starts, which should be 21 per neuron, 21*10. The total array size should be (784+1)*20+(20+1)*10, which is 15910
  so if I access index 15721, that's the 2nd layer, bias for the first neuron
  */
  // why even bother with a 1d array, seems complex? Speed. Because. IDK.

  //THE FIRST ELEMENT OF EACH ROW IS THE ANSWER, DONT COMPUTE IT

  /*******
  * MAIN *
  ********/
  public static void main(String[] args) {
    nn.menu();
  }

  /*******************
  * Statitical output*
  ********************/

  static void printStats(){
    String stats = "";
    int sumAllCorrect = 0;
    int sumAllTotal = 0;
    for(int i = 0; i < total.length; i++){
      stats += "For "+i+":"+correct[i]+"/"+total[i]+"\n";
      sumAllCorrect += correct[i];
      sumAllTotal +=total[i];
    }
    stats += "For all:" + sumAllCorrect + "/" + sumAllTotal + "\n";
    double correctPercent = (double)sumAllCorrect/(double)sumAllTotal;
    stats += " or " + correctPercent + "% accuracy";
    System.out.println(stats);
  }

  static void resetStats(){
    Arrays.setAll(correct, i -> { return 0; }); //reset all values to 0
    Arrays.setAll(total, i -> { return 0; }); 
  }


  /*******************************
  * Store and Load datasets/nets *
  ********************************/
  //done
  static void parseDataset(String datasetURI, int datasetSize) {
    String line = ""; //instantiation for later, this will hold each full line of pixel values
    dataset = new ArrayList<String>(datasetSize); // each row is an image
    try(BufferedReader buffer = new BufferedReader(new FileReader(datasetURI))) { //try reading in the file
      int i = 0;
      while((line = buffer.readLine()) != null) {
        dataset.add(i, line); // each line is 1/60000 of the set
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
      // System.out.println(neurNet.length);
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
      neurNet[i] = 2 * Math.random() - 1;
    }
    isLoaded = true;
  }

  /*************************************
   * stochastic gradient descent stuff *
   *************************************/

  //one call per node per layer
  static void biasGradientCalc(int nodeIndex, int layerIndex){
    //node and layerIndex refer to current, and need to be +1 to be last layer indices
    double biasGradient = 0.0;
    int nodeNumber = nodeIndex + 1; //nodeNumberForCurrentlayer node(this should not change per function call)
    int nodeSize = layers[layerIndex-1] + 1;
    int layerStartIndex;
    //bias gradient index is layers[layerIndex] because of +1 element for bias/-1 for offset, so L1 node0 (indexes) is neurNet[784]
    if(layerIndex < 2) { //middle layer
      int finalLayerNodeSize = layers[layerIndex] + 1;//add layersize + bias
      int finalLayerNodeIndex = 0;//nodeIndex for last layer node(this will iterate)
      int finalLayerNodeNumber = finalLayerNodeIndex+1;//GOTO: thsi si effectively useless
      int finalLayerStartIndex = layers[layerIndex]*(layers[layerIndex-1]+1); //index at which the next layer begins, also useful for storage 
      layerStartIndex = 0;
      // System.out.println("layerIndex should be 1:"+layerIndex+"\n");
      // sum all weights*biases respective node n, in all the nodes of the next layer. Also multiply that weight by the bias
      // layers[layerIndex] is number of nodes
      double sumOfWeightsByBiasGradient = 0.0; //runnning sum of weight*bias per last layer node
      int finalLayerBiasGradientIndex = finalLayerStartIndex+(finalLayerNodeNumber*finalLayerNodeSize)-1;//index for bias gradient of current node of next layer
      int finalLayerWeightIndex = finalLayerStartIndex+(finalLayerNodeNumber*finalLayerNodeSize-finalLayerNodeSize)+nodeIndex;//index for this node's weight in the current node of the next layer
      while(finalLayerNodeIndex < layers[2]){ //while lastNode index < total nodes in lastLayer
        sumOfWeightsByBiasGradient += neurNet[finalLayerWeightIndex] * gradients[finalLayerBiasGradientIndex]; //haha !
        finalLayerWeightIndex += finalLayerNodeSize;
        finalLayerBiasGradientIndex += finalLayerNodeSize;
        finalLayerNodeIndex++;
      }
      gradients[layerStartIndex+(nodeNumber*nodeSize-1)] = sumOfWeightsByBiasGradient*layerOutputs[layerIndex-1][nodeIndex]*(1-layerOutputs[layerIndex-1][nodeIndex]);
      // if its node0 layer0, bias is at 784, or rather, layers[layerIndex-1]
    } else { // index 2
      layerStartIndex = layers[layerIndex-1]*(layers[layerIndex-2]+1);//good 15699

      biasGradient = (layerOutputs[layerIndex-1][nodeIndex]-oneHotVector[nodeIndex])*layerOutputs[layerIndex-1][nodeIndex]*(1-layerOutputs[layerIndex-1][nodeIndex]);//good
      gradients[layerStartIndex+(nodeNumber*nodeSize-1)] = biasGradient;//GOOD
    }
  }

  //once per node
  static void weightGradient(double[] inputImage, int nodeIndex, int layerIndex){
    double weightGradient;
    int nodeNumber = nodeIndex+1;
    int nodeSize = layers[layerIndex-1]+1;
    if(layerIndex < 2){ //middle layer
      //the diff is the output comes from the dataset
      int layerStartIndex = 0;
      double biasGradient = gradients[layerStartIndex+(nodeNumber*nodeSize)-1];
      //GOTO: this is the PROBLEM^^^^^^^^^
      for(int imagePixelIndex = 0;imagePixelIndex < layers[layerIndex-1]; imagePixelIndex++){ //until through all pixels
        weightGradient = inputImage[imagePixelIndex] * biasGradient;
        gradients[layerStartIndex+(nodeNumber*nodeSize-nodeSize)+imagePixelIndex] = weightGradient;//start+(xn-n)+nodeweightindex
      }
    } else { //last layer
      int layerStartIndex = layers[layerIndex-1]*(layers[layerIndex-2]+1);
      int prevLayerNodeIndex = 0; //iterated
      double biasGradient = gradients[layerStartIndex+(nodeNumber*nodeSize-1)];
      while(prevLayerNodeIndex < layers[layerIndex-1]){ //loop thru prev nodes to get outputs(activations)
        weightGradient = layerOutputs[layerIndex-2][prevLayerNodeIndex] * biasGradient;
        gradients[layerStartIndex+(nodeNumber*nodeSize-nodeSize)+prevLayerNodeIndex] = weightGradient;//start+(xn-n)+nodeweightindex
        prevLayerNodeIndex++;
      }
    }
  }

  //once per node, but a lot: nodeweightssum*batchsize
  static void newWeight(int nodeIndex, int layerIndex){
    double learningGradient = learningRate/(double)batchSize;
    int nodeNumber = nodeIndex+1;
    int nodeSize = layers[layerIndex-1]+1;
    int layerStartIndex;
    if(layerIndex < 2){ //index 1
      layerStartIndex = 0;
    } else { //last layer, index 2
      layerStartIndex = layers[layerIndex-1]*(layers[layerIndex-2]+1);
    }
    for(int weightPrevLayerNodeToCurrNode = 0; weightPrevLayerNodeToCurrNode < layers[layerIndex-1]; weightPrevLayerNodeToCurrNode++){ //iterate through all weights on this node
      double weightGradientSum = 0.0; //sum weight gradient through all runs
      for(int batchRun = 0; batchRun < batchSize; batchRun++){//sum weight gradients
        weightGradientSum += gradientsPerBatch[batchRun][layerStartIndex+(nodeNumber*nodeSize-nodeSize)+weightPrevLayerNodeToCurrNode];
      }
      neurNet[layerStartIndex+(nodeNumber*nodeSize-nodeSize)+weightPrevLayerNodeToCurrNode] -= learningGradient * weightGradientSum;
    }
  }

  static void newBias(int nodeIndex, int layerIndex){
    double learningGradient = learningRate/(double)batchSize;
    int nodeNumber = nodeIndex+1;
    int nodeSize = layers[layerIndex-1]+1;
    int layerStartIndex;
    double biasGradientSum = 0.0; //for a single bias through all runs
    if(layerIndex < 2){ //index 1
      layerStartIndex = 0;
    } else { //last layer, index 2
      layerStartIndex = layers[layerIndex-1]*(layers[layerIndex-2]+1);
    }
    for(int batchRun = 0; batchRun < batchSize; batchRun++){//sum weight gradients
      biasGradientSum += gradientsPerBatch[batchRun][layerStartIndex+(nodeNumber*nodeSize-1)];
    }
    neurNet[layerStartIndex+(nodeNumber*nodeSize-1)] -= learningGradient * biasGradientSum;
  }

  /**************
   * batches area *
   **************/

  static void runBatches(){
    // shuffle data, pass batchSize at a time to rows
    // after finishing the initial bits, parameterize it
    Collections.shuffle(dataset); //randomizes dataset order
    int currIndex = 0;
    while(currIndex < dataset.size()){
      runBatch(currIndex);
      currIndex+=batchSize;
    }
  }

  static void runBatch(int currIndex){
    //call runNet with successive rows
    for(int batchRun = 0; batchRun < batchSize; batchRun++){
      runNet(currIndex, batchRun);
      currIndex++;
    }
    //TODO: update new weight/biases here
    // System.out.println("layeroutputs.length is:"+layerOutputs.length+"\n");
    if(!testing){
      for(int layerIndex = 1; layerIndex < layers.length; layerIndex++){ //increment through 2 layers
        for(int nodeIndex = 0; nodeIndex < layers[layerIndex]; nodeIndex++ ){ //increment through 
          newWeight(nodeIndex, layerIndex);
          newBias(nodeIndex, layerIndex);
        }
      }
    }
  }

  static double[] convertRowToDouble(String row){
    return Arrays.stream(row.split(",")).mapToDouble(Double::parseDouble).toArray(); //does some cool java 8 stream processing
  }


   /**********************
  *  NN running functions  *
  ***********************/

  static void runNet(int rowIndex, int batchRun){
    //init the layer lengths in jagged array
    double[] row = convertRowToDouble(dataset.get(rowIndex)); //entire single mnist row
    double[] inputImage = new double[layers[0]]; //image without correctNumber
    int correctNumber = (int)row[0];
    System.arraycopy(row, 1, inputImage, 0, layers[0]);
    //normalize greyscale
    // for(int i = 0; i < inputImage.length; i++){
    //   inputImage[i] /= 255.0; //assignment operators are the devil, and I am Faust
    //   // System.out.println(inputImage[i]);
    // }
    //layer 1
    layerOutputs[0] = new double[layers[1]]; //layerOutputs is layers-1 since we don't need to store L0
    layerActivation(1, inputImage);
    //additional layers(should run once for 3 layers)
    for(int layerIndex = 2; layerIndex < layers.length; layerIndex++){
      layerOutputs[layerIndex-1] = new double[layers[layerIndex]]; //layerOutputs is layers-1 since we don't need to store L0
      layerActivation(layerIndex, layerOutputs[layerIndex-2]); //pass in the current layer, as well as the output from the previous layer as input (-1 for prev layer, -1 to account for no layer 0)
    }
    setOneHotVector(correctNumber);
    updateTotals(correctNumber);
    //TODO: I don't think these for loops are correct
    //bias gradient before weight, bc weight uses bias
    //seperate loops for the same reason
    for(int layerIndex = 2; layerIndex > 0; layerIndex--) { //backprop, so decrement
      for(int nodeIndex= 0; nodeIndex < layers[layerIndex]; nodeIndex++){ //increment thru current layer nodes
        biasGradientCalc(nodeIndex, layerIndex);
      }
    }
    for(int layerIndex = 2; layerIndex > 0; layerIndex--) { //backprop, so decrement
      for(int nodeIndex = 0; nodeIndex < layers[layerIndex]; nodeIndex++){ //increment thru current layer nodes
        weightGradient(inputImage, nodeIndex, layerIndex);
      }
    }
    //store gradients for batch run, reset
    gradientsPerBatch[batchRun] = gradients;//store run for adjustments post-batch
  }

  static void setOneHotVector(int correctNumber){
    Arrays.setAll(oneHotVector, i -> { return 0; }); //flatten array
    oneHotVector[correctNumber] = 1;
  }

  static void updateTotals(int correctNumber) {
    int selectedByNet = 0;
    double bestSoFar = 0.0;
    //find the highest value
    for(int i = 0; i < layers[layers.length-1]; i++){
      // System.out.println("best is"+bestSoFar+" and layerOtputs"+layerOutputs[1][i]+"and correctnumber is"+correctNumber+"i is"+i+"\n");
      if(layerOutputs[1][i] > bestSoFar){
        bestSoFar = layerOutputs[1][i];
        selectedByNet = i;
      }
    }
    nn.total[correctNumber]+=1; //increase the total for the number
    if(correctNumber == selectedByNet){
      nn.correct[correctNumber]++; //increase correct count for number
    }
  }
 
  static void layerActivation(int layerIndex, double[] inputs) {
    // split out the current layer from the network
    //we only have 2 layers, so layer 1 and layer 2
    //layer 0 is just the base inputs
    int srcPos, destPos, length;
    //assume layerIndex is never less than 1 bc 0 is just inputs(no calc needed)
    //works for any nnumber of layers(?)
    if(layerIndex > 1){ //TODO: could simplify this logic if java does truthiness stuff
      srcPos = (nn.layers[layerIndex-1]*(nn.layers[layerIndex-2]+1)); //layer 1 nodes * weights + bias(elements per node) for layer 0 gives index of last element
      destPos = 0;
      length = nn.layers[layerIndex]*(nn.layers[layerIndex-1]+1);
      // int ending = length+srcPos-1;
      // System.out.println("start:"+srcPos+"end:"+ending+"size:"+length+"\n");//should be 15700 and 15909
    } else {
      srcPos = 0; //layer 1 starts at 0
      destPos = 0;
      length = (nn.layers[1]*(nn.layers[0]+1)); //last index before layer 2 starts
      // int ending = length+srcPos-1;
      // System.out.println("start:"+srcPos+"end:"+ending+"size:"+length+"\n"); //should be 0 and 15699
    }
    double[] layerWeightsNBiases = new double[length];  //this should be an entire set of weights+a bias for a layer
    System.arraycopy(neurNet, srcPos, layerWeightsNBiases, destPos, length); //grabs layer subset
    //TODO: TBH try not to do this, replace with specific index accesses instead, otherwise defeats purpose of single array
    for(int nodeIndex = 0; nodeIndex < nn.layers[layerIndex]; nodeIndex++) { //on layer 1, runs 20 times 
      int nodeNumber = nodeIndex+1; //this is fine
      int numberOfElementsForANode = nn.layers[layerIndex-1]+1; //inputs from prev layer + bias
      double[] weights = new double[nn.layers[layerIndex-1]];//set weights == number outputs previous layer
      System.arraycopy(layerWeightsNBiases, nodeIndex*(nn.layers[layerIndex]+1), weights, 0, nn.layers[layerIndex-1]);// 0*20, 1*20(start at 20th index, makes sense since 0 indexed offset ends at 20)
      //source: 785 elements per node, 
      //destination: 784 elements, end index is one before source
      //need to skip over bias, first node weights go to 783, bias at 784, second node starts at 785
      double bias = layerWeightsNBiases[nodeNumber*numberOfElementsForANode-1]; // the first bias is at element 785, nodeNumber*(numberOfWeightsForNode), 1*785=785, but the index is -1
      //node 1: 1*785-1= index 784 : correct
      //node 2: 2*785-1= index 1569 : correct
      //node 3: 3*785-1= index 2354 : correct
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
    double sigmoid = 1/(1 + Math.exp(-(preSigmoid+bias)));
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
        if(!isLoaded){
          initNet();
        }
        testing = false;
        parseDataset(nnTraining, 60000);
        runBatches();
        printStats();
        resetStats();
        break;
      case "2":
        isLoaded = nn.loadNet();
        break;
      case "3": //test against training data
        if(isLoaded) {
          testing = true;
          nn.parseDataset(nnTraining, 60000);
          runBatches();
          printStats();
          resetStats();
        }
        break;
      case "4": //test against testing data
        if(isLoaded) {
          testing = true;
          nn.parseDataset(nnTest, 60000);
          runBatches();
          printStats();
          resetStats();
        }
        break;
      case "5": //save net
        if(isLoaded) {
          nn.storeNet();
        }
        break;
      case "6": //run epoch times
        if(isLoaded) {
          for(int i = 0; i< epochsToRun;i++){
            runBatches();
          }
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