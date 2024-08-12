"""
Assignment03:快速排序
时间复杂度：O（nlogn）
"""
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def quick_sort(nums, start, end):
            left, right = start, end
            if left >= right:
                return
            mid = random.randint(left, right)
            nums[left], nums[mid] = nums[mid], nums[left]
            pivot = nums[left]
            while left < right:
                while left < right and nums[right] >= pivot:
                    right -= 1
                nums[left] = nums[right]
                while left < right and nums[left] < pivot:
                    left += 1
                nums[right] = nums[left]
            nums[left] = pivot
            while left >= 0 and nums[left] == pivot:
                left -= 1
            while right < len(nums) and nums[right] == pivot:
                right += 1
            quick_sort(nums, start, left)
            quick_sort(nums, right, end)

        quick_sort(nums, 0, len(nums) - 1)
        return nums