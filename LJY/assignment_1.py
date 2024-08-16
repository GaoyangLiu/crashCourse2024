def search(nums,target):

    left , right = 0 , len(nums) - 1

    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid -1

    return -1

def main():
    arr_input = input("输入：nums = ")
    nums = [int(x) for x in arr_input.strip('[]').split(',')]

    target = int(input("输入： target = "))

    result = search(nums, target)
    if result != -1:
        print(f"{target}出现在nums中并且下标为{result}")
    else:
        print(f"{target}不存在nums中因此返回-1")

    input("按任意键退出...")

if __name__=="__main__":
    main()