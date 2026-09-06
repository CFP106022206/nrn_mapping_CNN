"""Neuron matching service package.

前端上傳一個 SWC -> 找出另一個資料庫中最相似的 n 個神經元。

流程與離線 pipeline 完全共用同一組參數（見 config.py），
所以服務端算出來的三視圖和資料庫裡歸檔的三視圖在同一個尺度上。
"""
