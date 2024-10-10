class Solution(object):

    def sortArray(self, nums):

        """:type nums: List[int] :rtype: List[int]"""



        minval, maxval = min(nums), max(nums)#找出数组中的最大值、最小值

        length = maxval - minval + 1#计算新数组长度

        counts = [0] * length;

        for num in nums:

            #把每个元素通过偏移量调整到0-length

            counts[num - minval] += 1

        j = 0

        for i in range(length):

            while counts[i] > 0:

                nums[j] = i + minval

                counts[i] -= 1

                j += 1

            #对有多个重复的整数尾插到nums中

        return nums