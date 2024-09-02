
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const { parse } = require('json2csv');
const csvjson = require('csvjson');
const axios = require('axios');

const app = express();
const port = 8000;

let uploadedData = [];
let csvData;
let filePath = null;


app.use(express.static(path.join(__dirname, 'public')));


app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));


const fileFilter = (req, file, cb) => {
    if (file.mimetype === 'text/csv' || file.mimetype === 'application/vnd.ms-excel') {
        cb(null, true);
    } else {
        cb(new Error('Only CSV files are allowed'), false);
    }
};

const upload = multer({
    
    fileFilter: fileFilter,
    dest: 'uploads/'

});


app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'Home.html'));
});


app.get('/upload.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'upload.html'));
});


app.get('/display.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'display.html'));
});


app.post('/upload', upload.single('fileToUpload'), (req, res) => {

    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }
    
    filePath = path.join(__dirname, req.file.path);
    
    fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (row) => {
            uploadedData.push(row);
        })
        .on('end', () => {
            console.log('CSV file successfully processed');
            res.redirect('/display.html');
        });
});

app.get('/data', (req, res) => {
    res.json(uploadedData);
});

app.post('/train', async (req, res) => {
    if (!filePath) {
        return res.status(400).send('No CSV file uploaded yet!');
    }
    try {
        const csvData = await fs.promises.readFile(filePath, 'utf-8');
        const url = 'http://localhost:5000/train';
        const response = await fetch(url, {
            method: 'POST',
            body: csvData,
            headers: {
                'Content-Type': 'text/csv'
            }                                                                                                                       
        });
        res.send(response);
    } catch (error) {
        console.error('Error processing CSV:', error);
        res.status(500).send('Error sending processing request'); // Send an error response
    }
});
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
