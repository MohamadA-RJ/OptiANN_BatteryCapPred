function cap = extract_discharge(B)
bcycle = B.cycles;
for i = 1:length(bcycle)
    if isequal(bcycle(i).type, 'discharging')
        cap(i) = bcycle(i).data.capacity;
    end
end
cap(cap==0) = [];