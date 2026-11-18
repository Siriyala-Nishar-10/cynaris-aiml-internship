
# W11D3 - MLflow REST API Test

$columns = @(
    "feature_0", "feature_1", "feature_2", "feature_3",
    "feature_4", "feature_5", "feature_6", "feature_7",
    "feature_8", "feature_9", "feature_10", "feature_11"
)

$row = @(
    0.5, -1.2, 1.0, 0.3,
    -0.7, 1.5, 0.2, -0.4,
    0.8, 1.1, -0.6, 0.9
)

$bodyObject = @{
    dataframe_split = @{
        columns = $columns
        data = ,$row
    }
}

$body = $bodyObject | ConvertTo-Json -Depth 10

Write-Host "`nSending request to MLflow REST API..."

try {
    $result = Invoke-RestMethod `
        -Uri "http://127.0.0.1:5003/invocations" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body

    Write-Host "`n========== API TEST SUCCESS =========="
    $result | ConvertTo-Json -Depth 10
}
catch {
    Write-Host "`n========== API TEST FAILED =========="
    Write-Host "Status Code:"
    Write-Host ([int]$_.Exception.Response.StatusCode)

    Write-Host "`nMLflow Error Details:"
    Write-Host $_.ErrorDetails.Message
}
