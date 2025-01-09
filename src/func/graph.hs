{-
data Set a = S [a]

elem :: a -> Set a -> Bool
elem x (S xs) = x == head xs || elem x (S (tail xs))

add :: a -> Set a -> Set a
add x s@(S xs)
    | elem x s = s
    | otherwise = S (x : xs)

union :: Set a -> Set a -> Set a
union (Set xs) (Set ys) = if elem (head ys) (Set xs) then union (Set xs) (Set (tail ys)) else union (Set ((head ys) : xs)) (Set (tail ys))

intersect :: Set a -> Set a -> Set a
intersect x y 

setminus :: Set a -> Set a -> Set a 


data Graph = G [[Int]] | E Set
-}
