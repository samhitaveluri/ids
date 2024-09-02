function updateLabel() {
    const fileInput = document.getElementById('fileToUpload');
    const fileLabel = document.getElementById('fileLabel');

    if (fileInput.files.length > 0) {
        const fileName = fileInput.files[0].name;
        fileLabel.textContent = fileName;
    }
}
function handleFile() {
    const fileInput = document.getElementById('fileToUpload');
    const file = fileInput.files[0];

    if (!file) {
        alert('No file selected.');
        return;
    }

    if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file.');
        // Clear the file input field
        fileInput.value = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = function(event) {
        const contents = event.target.result;
        // Encode the CSV data to be passed as a query parameter
        window.location.href = 'dataset.html';
        // Redirect the user to another webpage with the CSV data as a query parameter
    };
    
    

    reader.onerror = function(event) {
        console.error('File reading error:', event.target.error);
    };

    reader.readAsText(file);
}
