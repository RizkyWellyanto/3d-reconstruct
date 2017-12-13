%% Return iminfo struct containing 10 fields of N length, where N is total amount of CVPR images/data
function iminfo = generate_iminfo(paths)
    for i = 1:numel(paths)
        init_iminfo = CreateImInfo(paths{i}); 
        full_iminfo = parse_iminfo(init_iminfo);

        if i > 1
            iminfo = struct_concat(iminfo, full_iminfo);
        else
            iminfo = full_iminfo;
        end
    end
end