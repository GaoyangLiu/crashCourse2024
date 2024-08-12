"""
Assignment02:归并排序
时间复杂度：O（nlogn）
"""
# 归并排序
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def mergeSort(arr, low, high):
            if low >= high:                 
                return

            mid = low + (high-low)//2       
            mergeSort(arr, low, mid)        
            mergeSort(arr, mid+1, high)

            left, right = low, mid+1        
            tmp = []                        
            while left <= mid and right <= high:    
                if arr[left] <= arr[right]:         
                    tmp.append(arr[left])
                    left += 1
                else:                              
                    tmp.append(arr[right])
                    right += 1
            
            while left <= mid:            
                tmp.append(arr[left])
                left += 1
          
            while right <= high:            
                tmp.append(arr[right])
                right += 1
          
            arr[low: high+1] = tmp          
        
        mergeSort(nums, 0, len(nums)-1)     
        return nums