const NeuralNetwork = require("./NeuralNetwork");
const NeuralNetworkV2 = require("./NeuralNetworkV2");
const _ = require('lodash');


//init data
const trainData = [
    {input: [0, 0], output: 0},
    {input: [1, 0], output: 1},
    {input: [0, 1], output: 1},
    {input: [1, 1], output: 0},
];

const neuralNetworkV1 = new NeuralNetwork();
const neuralNetwork = new NeuralNetworkV2();

console.log('before training');
console.log(`0 XOR 0 => ${neuralNetwork.calcResult(0, 0)}`);
console.log(`1 XOR 1 => ${neuralNetwork.calcResult(1, 1)}`);

console.log('training');
console.log('epoch 1 - with 1000 times');
_.times(1000, () => {
    neuralNetwork.trainNeural(trainData);
});


console.log('epoch 2 - with 1000 times')
_.times(1000, () => {
    neuralNetwork.trainNeural(trainData);
});

console.log('--------------------');
console.log('working');
console.log(`0 XOR 0 => ${neuralNetwork.calcResult(0, 0)}`);
console.log(`0 XOR 0 => ${neuralNetwork.calcResult(0, 0)}`);
console.log(`1 XOR 0 => ${neuralNetwork.calcResult(1, 0)}`);
console.log(`1 XOR 0 => ${neuralNetwork.calcResult(1, 0)}`);
console.log(`1 XOR 1 => ${neuralNetwork.calcResult(1, 1)}`);
