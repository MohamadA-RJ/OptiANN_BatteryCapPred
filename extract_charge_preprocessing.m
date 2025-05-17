function charInput = extract_charge_preprocessing(B)
bcycle = B.cycles;
for i = 1:length(bcycle)-1
    if isequal(bcycle(i).type, 'charging')
        le = mod(length(bcycle(i).data.Voltage_measured), 10);
        vTemp = bcycle(i).data.Voltage_measured(:, 1:end-le);
        vTemp = reshape(vTemp, length(vTemp)/10, []);
        vTemp = mean(vTemp);
        
        iTemp = bcycle(i).data.Current_measured(:, 1:end-le);
        iTemp = reshape(iTemp, length(iTemp)/10, []);
        iTemp = mean(iTemp);

        charInput(i, :) = [vTemp, iTemp, ];
    end
end
% Remove any rows with all zero elements
    charInput(~any(charInput, 2), :) = [];
end