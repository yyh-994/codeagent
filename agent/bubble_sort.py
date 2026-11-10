"""冒泡排序实现。"""


def bubble_sort(nums):
    """返回升序排列的新列表，不修改原列表。"""
    arr = list(nums)
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # 已有序，提前结束
            break
    return arr


if __name__ == "__main__":
    data = [5, 2, 9, 1, 5, 6]
    print("排序前:", data)
    print("排序后:", bubble_sort(data))
