// get data from form
$(document).ready(function () {
	function handleInputChange() {
		var bulan = $("#bulan").val();
		var tahun = $("#tahun").val();

		var csvFilePath = "/static/data.csv";

		$.ajax({
			type: "GET",
			url: csvFilePath,
			dataType: "text",
			success: function (data) {
				processData(data, bulan, tahun);
			},
		});
	}

	$("#bulan, #tahun").change(handleInputChange);
});

// Fungsi proses data CSV
function processData(csvData, bulan, tahun) {
	var rows = csvData.split("\n");
	rows.forEach(function (row) {
		var columns = row.split(",");
		if (columns[1] == tahun && columns[0] == bulan) {
			minyak = columns[2];
			dolar = columns[3];
			// change value of input #minyak and #dolar
			$("#minyak").val(minyak);
			$("#dollar").val(dolar);
		}
	});
}
