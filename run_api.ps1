$filePath = "list-api.md"
$lines = Get-Content -Path $filePath

Write-Host "Memulai proses download data NASA POWER..." -ForegroundColor Green

foreach ($line in $lines) {
    $cleanLine = $line.Trim()
    
    if ($cleanLine.StartsWith("curl ")) {
        # Mengubah kutip tunggal (') menjadi kutip ganda (") agar simbol & aman di Windows
        $safeLine = $cleanLine -replace "'", '"'
        
        $fileName = ""
        if ($cleanLine -match "-o\s+(.*\.csv)") {
            $fileName = $matches[1]
        }

        Write-Host "Mengunduh: $fileName" -ForegroundColor Cyan
        
        # Eksekusi menggunakan safeLine
        cmd.exe /c $safeLine
        
        Write-Host "Menunggu 3 detik..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 3
    }
}

Write-Host "Semua proses unduhan selesai!" -ForegroundColor Green