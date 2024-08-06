
def sortArray(nums):
    if len(nums) <=1:
       return nums

    mid = len(nums) //2
    left_half = nums[:mid]
    right_half = nums[mid:]

    left_sorted = sortArray(left_half)
    right_sorted = sortArray(right_half)

    return merge(left_sorted, right_sorted)

def merge(left , right):
    sorted_arr = []
    left_index = 0
    right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            sorted_arr.append(left[left_index])
            left_index += 1
        else:
            sorted_arr.append(right[right_index])
            right_index += 1

    while left_index < len(left):
        sorted_arr.append(left[left_index])
        left_index +=1

    while right_index < len(right):
        sorted_arr.append(right[right_index])
        right_index +=1

    return sorted_arr

def main():
    arr_input = input("输入：nums = ")
    nums = [int(x) for x in arr_input.strip('[]').split(',')]
    result = sortArray(nums)
    print(result)

    input("按任意键退出...")

if __name__=="__main__":
    main()