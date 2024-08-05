class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        nums.sort()
        return nums


b = Solution()
print(b.sortArray([5, 1, 1, 2, 0, 0]))


class Solution1:
    def sortArray(self, nums: list[int]) -> list[int]:
        for i in range(len(nums)):
            for j in range(len(nums) - i - 1):
                if nums[j + 1] < nums[j]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        return nums

c = Solution1()
print(c.sortArray([5, 1, 1, 2, 0, 0]))
