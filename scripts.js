let net;

const webCamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const descripcionVideo = document.getElementById('descripcion_video')

const app = async () => {
    net = await mobilenet.load()

    webCam = await tf.data.webcam(webCamElement);
    while(true){

        const img = await webCam.capture();
        const result = await net.classify(img);

        const activation = net.infer(img, "conv_preds");

        var result2;
        try{
            result2 = await classifier.predictClass(activation);
            const billetes = ["Untrained","20","50","100","500","1000"]
            document.getElementById('prediction2').innerHTML = "trainning prediction: " + billetes[result2.label];

        }catch(error){
            console.log("imagen no registrada: "+ error)
        }

        document.getElementById('prediction').innerHTML = 'prediction: ' + result[0].className;
        document.getElementById('probability').innerHTML = "probability: " + result[0].probability;
        
        img.dispose();
        await tf.nextFrame();
    }
}

async function addExample(classId){
    const img = await webCam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);
    
    console.log("ejemplo agregado")

    img.dispose();

}

const saveKnn = async () => {
    //obtenemos el dataset actual del clasificador (labels y vectores)
    let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    const storageKey = "knnClassifier";
    //lo almacenamos en el localStorage
    localStorage.setItem(storageKey, strClassifier);
};


const loadKnn = async ()=>{
    const storageKey = "knnClassifier";
    let datasetJson = localStorage.getItem(storageKey);
    classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
};


app()