class Solution(object):
    def sortArray(self, nums):
        '''
        归并排序：
        利用递归过程。将原数组分为左右两个，
        左边右边分别有序后，再分别指针从左走，然后比较左右的数
        依次传进辅助数组中，直到有一侧赋值完以后，另一侧的剩下的直接拷贝
        当全部完成以后，将辅助数组中的数全部拷贝原数组即可
        '''
        if len(nums) < 2:
            return nums
        return self.sortProcess(nums, 0, len(nums)-1)

    def sortProcess(self, nums, L, R):
        '''
        归并排序中的排序功能函数

        递归的时候参数不能出现常数，因为参数就是不断地变化的，如果定死了常数就不变了
        '''
        if L == R:
            return

        mid = L + ((R-L) >> 1)   #相当于（L+R）//2
        self.sortProcess(nums, L, mid)
        self.sortProcess(nums, mid+1, R)  #这块就已经做到了左右两边分别有序
        return self.merge(nums, L, mid, R)

    def merge(self, nums, L, mid, R):
        '''
        归并排序中的归并函数功能
        即将两个排序好的子序列合并到一起

        '''
        help_length = R - L + 1
        help_nums = []

        for i in range(0, help_length):
            help_nums.append(0)

        index = 0 #辅助数组的索引
        p1 = L   # L和R是随着变化的，不能一直赋值为0
        p2 = mid+1
        while p1 <= mid and p2 <= R : #保证不越界，也就是给一个终止条件
            if (nums[p1] < nums[p2]) :
                help_nums[index] = nums[p1]
                index += 1
                p1 += 1
            else :
                help_nums[index] = nums[p2]
                index += 1
                p2 += 1

        while p1 <= mid:
            help_nums[index] = nums[p1]
            p1 += 1
            index += 1


        while p2 <= R:
            help_nums[index] = nums[p2]
            p2 += 1
            index += 1


        for i in range (len(help_nums)):
            nums[L+i] = help_nums[i]

        return nums
