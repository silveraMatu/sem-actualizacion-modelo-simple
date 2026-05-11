const TOTAL_EPOCHS = 350;

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const epochInfo = document.getElementById('epoch-info');
const progressBar = document.getElementById('progress');
const predictionPanel = document.getElementById('prediction-panel');
const xInput = document.getElementById('x-input');
const predictBtn = document.getElementById('predict-btn');
const resultContainer = document.getElementById('result-container');
const resultValue = document.getElementById('result-value');

async function run() {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xsData = [-6, -5, -4, -3, -2, -1, 0, 1, 2];
    const ysData = xsData.map(x => 2 * x + 6);
    
    const xs = tf.tensor2d(xsData, [9, 1]);
    const ys = tf.tensor2d(ysData, [9, 1]);

    await model.fit(xs, ys, {
        epochs: TOTAL_EPOCHS,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const currentEpoch = epoch + 1;
                epochInfo.innerText = `Época: ${currentEpoch} / ${TOTAL_EPOCHS}`;
                
                const progressPercentage = (currentEpoch / TOTAL_EPOCHS) * 100;
                progressBar.style.width = `${progressPercentage}%`;
                
                if (currentEpoch % 50 === 0 || currentEpoch === TOTAL_EPOCHS) {
                    console.log(`Epoch ${currentEpoch}: loss = ${logs.loss}`);
                }
            }
        }
    });

    trainingComplete();

    setupPrediction(model);
}

function trainingComplete() {
    statusDot.classList.remove('training');
    statusDot.classList.add('ready');
    statusText.innerText = 'Modelo Listo';
    statusText.style.color = 'var(--success)';
    epochInfo.innerText = `Entrenamiento finalizado (${TOTAL_EPOCHS} épocas)`;
    
    predictionPanel.classList.remove('hidden');
    
    xInput.disabled = false;
    predictBtn.disabled = false;
    xInput.focus();
}

function setupPrediction(model) {
    predictBtn.addEventListener('click', () => {
        const xValue = parseFloat(xInput.value);
        
        if (isNaN(xValue)) {
            alert('Por favor, ingresa un número válido para X.');
            return;
        }

        const output = model.predict(tf.tensor2d([xValue], [1, 1]));
        const prediction = output.dataSync()[0];
        
        resultContainer.classList.remove('hidden');
        
        animateValue(resultValue, prediction, 1000);
    });

    xInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !predictBtn.disabled) {
            predictBtn.click();
        }
    });
}

function animateValue(obj, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        
        const easeProgress = 1 - Math.pow(1 - progress, 4);
        const currentVal = end * easeProgress;
        
        obj.innerHTML = currentVal.toFixed(2);
        
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.innerHTML = end.toFixed(2);
        }
    };
    window.requestAnimationFrame(step);
}

document.addEventListener('DOMContentLoaded', run);