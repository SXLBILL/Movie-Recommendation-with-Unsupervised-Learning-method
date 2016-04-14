function [user2, mov2] = transform(user, mov, cstMatch, movMatch)
    user2 = find(cstMatch(:,1)==user);
    mov2 = find(movMatch(:,1)==mov);
end

