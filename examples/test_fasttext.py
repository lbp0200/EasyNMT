from ftlangdetect import detect

result = detect(text="薄雾", low_memory=False)
print(result)

# result = detect(text="Bugün hava çok güzel", low_memory=True)
# print(result)