class Solution:
    def search(self, nums: list[int], target: int) -> int:
        for i in nums:
            if i == target:
                return (nums.index(i))
        return -1


c1 = Solution()
c2 = Solution()
print(c1.search(nums=[-1, 0, 3, 5, 9, 12], target=9))
print(c2.search(nums=[-1, 0, 3, 5, 9, 12], target=2))


class Solution1:
    def search(self, nums: list[int], target: int) -> int:
        i = 0
        j = len(nums) - 1
        while i <= j:
            m = i + (j-i) // 2
            if nums[m] < target:
                i = m + 1
            elif nums[m] > target:
                j = m - 1
            else:
                return m
        return -1


c3 = Solution1()
c4 = Solution1()
print(c3.search(nums=[-1, 0, 3, 5, 9, 12], target=9))
print(c4.search(nums=[-1, 0, 3, 5, 9, 12], target=2))
