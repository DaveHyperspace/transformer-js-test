import { pipeline, env } from '@xenova/transformers';
import { AutoModel, AutoTokenizer } from '@xenova/transformers';

// get path dir
import path from 'path';
const __dirname = path.resolve();

const modelPath = path.join(__dirname, '/models/');
const wasmPath = path.join(__dirname, '/wasm/');

env.localModelPath = './models/';

// Disable the loading of remote models from the Hugging Face Hub:
env.allowRemoteModels = false;

// Set location of .wasm files. Defaults to use a CDN.
env.backends.onnx.wasm.wasmPaths = '../wasm/';
// // Allocate a pipeline for sentiment-analysis
// let pipe = await pipeline('sentiment-analysis', 'bert-base-uncased');

// let out = await pipe('I love transformers!');
// // [{'label': 'POSITIVE', 'score': 0.999817686}]

// console.log(out);

let tokenizer = await AutoTokenizer.from_pretrained('gpt2');
let model = await AutoModel.from_pretrained('gpt2');

let question = 'What is the capital of France?';
let inputs = await tokenizer(question);
let res = await model(inputs);
console.log(res);

// console.log({ test: logits });
