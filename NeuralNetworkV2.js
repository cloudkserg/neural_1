const random = require('seed-random')(1337);
const _ = require('lodash');
class NeuralNetworkV2 {
    constructor () {
        this.weights = {
            i1_h1: random(),
            i2_h1: random(),
            bias_h1: random(),
            i1_h2: random(),
            i2_h2: random(),
            bias_h2: random(),
            h1_o1: random(),
            h2_o1: random(),
            bias_o1: random()
        };
    }

    activationSigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    //производная
    derivativeSigmoid(x) {
        const y = this.activationSigmoid(x);
        return y * (1 - y);
    }

    calcResult(i1, i2, returnAllResults = false) {
        const h1_input = this.weights.i1_h1 * i1 + this.weights.i2_h1 * i2 + this.weights.bias_h1;
        const h1 = this.activationSigmoid(h1_input);

        const h2_input = this.weights.i1_h2 * i1 + this.weights.i2_h2 * i2 + this.weights.bias_h2;
        const h2 = this.activationSigmoid(h2_input);

        const o1_input = this.weights.h1_o1 * h1 + this.weights.h2_o1 * h2 + this.weights.bias_o1;
        const o1 = this.activationSigmoid(o1_input);
        //console.log({h1_input, h1, h2_input, h2, o1_input, o1});
        if (returnAllResults) {
            return {h1_input, h1, h2_input, h2, o1_input, o1};
        }
        return o1;
    }

    calcDifferentWithoutResult (neuralResult, waitedResult) {
        return Map.pow(waitedResult - neuralResult, 2);
    }


    trainNeuralEl (weightDeltas, trainEl) {
        const i1 = trainEl.input[0];
        const i2 = trainEl.input[1];
        //рассчитываем еще раз так как это важно
        const {h1_input, h1, h2_input, h2, o1_input, o1} = this.calcResult(i1, i2, true);

        //тренируемся - рассчитываем делту
        const outputDelta = trainEl.output - o1;
        //тренируемся - рассчитваем производнй
        const o1Delta = outputDelta * this.derivativeSigmoid(o1_input);
        //считаем дельты весов
        weightDeltas.h1_o1 += h1 * o1Delta;
        weightDeltas.h2_o1 += h2 * o1Delta;
        weightDeltas.bias_o1 += o1Delta;

        //рассчитываем произодные входных нейронов
        const h1Delta = o1Delta * this.derivativeSigmoid(h1_input);
        const h2Delta = o1Delta * this.derivativeSigmoid(h2_input);

        weightDeltas.i1_h1 += i1 * h1Delta;
        weightDeltas.i2_h1 += i2 * h1Delta;
        weightDeltas.bias_h1 += h1Delta;

        weightDeltas.i1_h2 += i1 * h2Delta;
        weightDeltas.i2_h2 += i2 * h2Delta;
        weightDeltas.bias_h2 += h2Delta;

        return weightDeltas;
    }

    trainNeural (trainData) {
        const initWeightDeltas = {
            i1_h1: 0,
            i2_h1: 0,
            bias_h1: 0,
            i1_h2: 0,
            i2_h2: 0,
            bias_h2: 0,
            h1_o1: 0,
            h2_o1: 0,
            bias_o1: 0
        };

        //рассчитали веса
        const weightDeltas = _.reduce(trainData, (weightDeltas, trainEl) => {
            return this.trainNeuralEl(weightDeltas, trainEl);
        }, initWeightDeltas);

        Object.keys(this.weights).forEach(weightKey => {
            this.weights[weightKey] += weightDeltas[weightKey];
        });
    }

}

module.exports = NeuralNetworkV2;
